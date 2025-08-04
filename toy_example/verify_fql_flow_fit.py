import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import matplotlib.pyplot as plt
from absl import flags
import sys
import os
from tqdm import tqdm
from scipy.stats import multivariate_normal
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agents.fql import FQLAgent

# GPU configuration
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup_gpu():
    """Configure JAX to use GPU."""
    print("Setting up GPU...")
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
    if gpu_devices:
        jax.config.update('jax_default_device', gpu_devices[0])
        print(f"✅ Using GPU: {gpu_devices[0]}")
    else:
        print("⚠️ No GPU found, using CPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    return gpu_devices

def generate_toy_data(batch_size, key):
    """Generate 4x4 grid distribution toy data."""
    # Create 4x4 grid centers
    grid_size = 4
    spacing = 2.0 / (grid_size * 2)
    start = -1.0 + spacing
    coords = [start + i * spacing * 2 for i in range(grid_size)]
    
    # Generate all grid points
    mus = []
    for x in coords:
        for y in coords:
            mus.append([x, y])
    mus = jnp.array(mus)
    
    sigma = 0.1
    weights = jnp.ones(grid_size * grid_size) / (grid_size * grid_size)
    
    # Sample from mixture
    cat_key, normal_key = jax.random.split(key)
    component_indices = jax.random.categorical(cat_key, jnp.log(weights), shape=(batch_size,))
    mu_selected = mus[component_indices]
    noise = jax.random.normal(normal_key, shape=(batch_size, 2)) * sigma
    samples = mu_selected + noise
    
    return jnp.clip(samples, -1.0, 1.0)

def compute_true_density(X, Y):
    """Compute true distribution density."""
    grid_size = 4
    spacing = 2.0 / (grid_size * 2)
    start = -1.0 + spacing
    coords = [start + i * spacing * 2 for i in range(grid_size)]
    
    mus = []
    for x in coords:
        for y in coords:
            mus.append([x, y])
    mus = np.array(mus)
    
    sigma = 0.1
    weights = np.ones(grid_size * grid_size) / (grid_size * grid_size)
    
    density = np.zeros_like(X)
    for mu, w in zip(mus, weights):
        rv = multivariate_normal(mu, sigma**2 * np.eye(2))
        density += w * rv.pdf(np.dstack([X, Y]))
    
    return density

def create_config():
    """Create FQL agent configuration."""
    config = ml_collections.ConfigDict()
    config.lr = 4e-4
    config.value_hidden_dims = (256, 256)
    config.actor_hidden_dims = (256, 256, 256, 256)
    config.layer_norm = False
    config.actor_layer_norm = False
    config.encoder = None
    config.flow_steps = 10
    config.alpha = 1.0
    config.tau = 0.005
    config.q_agg = 'min'
    config.discount = 0.99
    config.normalize_q_loss = False
    return config

def create_observations(batch_size, obs_dim):
    """Create observation data."""
    return jnp.ones((batch_size, obs_dim))

def setup_flags():
    """Setup necessary flags."""
    if not hasattr(flags.FLAGS, 'offline_steps'):
        flags.DEFINE_integer('offline_steps', 10000, 'Number of offline training steps')
    if not hasattr(flags.FLAGS, 'pretrain_factor'):
        flags.DEFINE_float('pretrain_factor', 1.0, 'Pretrain factor')
    if not hasattr(flags.FLAGS, 'online_steps'):
        flags.DEFINE_integer('online_steps', 0, 'Number of online training steps')
    
    try:
        flags.FLAGS(['verify_fql_flow_fit.py'])
    except flags.DuplicateFlagError:
        pass

def create_agent(config, obs_dim, action_dim, seed):
    """Create FQL agent."""
    print("🤖 Creating FQL agent...")
    setup_flags()
    
    ex_observations = create_observations(1000, obs_dim)
    ex_actions = jnp.zeros((1000, action_dim))
    
    agent = FQLAgent.create(
        seed=seed,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config
    )
    print("✅ Agent created successfully")
    return agent

def flow_only_loss(agent, batch, grad_params, rng):
    """Compute flow-only loss for FQL."""
    batch_size, action_dim = batch['actions'].shape
    rng, x_rng, t_rng, sample_rng = jax.random.split(rng, 4)

    # BC flow loss
    x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
    x_1 = batch['actions']
    t = jax.random.uniform(t_rng, (batch_size, 1))
    x_t = (1 - t) * x_0 + t * x_1
    vel = x_1 - x_0

    pred = agent.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
    bc_flow_loss = jnp.mean((pred - vel) ** 2)

    return bc_flow_loss, {'flow_loss': bc_flow_loss}

def compute_2d_wasserstein_distance(learned_samples, true_samples):
    """Compute 2D Wasserstein distance using optimal transport."""
    try:
        import ot
        
        # Filter samples to [-1, 1] range
        learned_mask = (learned_samples[:, 0] >= -1.0) & (learned_samples[:, 0] <= 1.0) & \
                      (learned_samples[:, 1] >= -1.0) & (learned_samples[:, 1] <= 1.0)
        true_mask = (true_samples[:, 0] >= -1.0) & (true_samples[:, 0] <= 1.0) & \
                   (true_samples[:, 1] >= -1.0) & (true_samples[:, 1] <= 1.0)
        
        learned_filtered = learned_samples[learned_mask]
        true_filtered = true_samples[true_mask]
        
        # Subsample for efficiency
        max_samples = 1000
        if len(learned_filtered) > max_samples:
            idx = np.random.choice(len(learned_filtered), max_samples, replace=False)
            learned_filtered = learned_filtered[idx]
        if len(true_filtered) > max_samples:
            idx = np.random.choice(len(true_filtered), max_samples, replace=False)
            true_filtered = true_filtered[idx]
        
        # Compute optimal transport
        a = np.ones(len(learned_filtered)) / len(learned_filtered)
        b = np.ones(len(true_filtered)) / len(true_filtered)
        M = ot.dist(learned_filtered, true_filtered, metric='euclidean')
        
        emd_value = ot.emd2(a, b, M)
        return np.sqrt(emd_value)
        
    except ImportError:
        print("⚠️ Python Optimal Transport (ot) library not available")
        return None
    except Exception as e:
        print(f"⚠️ Optimal Transport computation failed: {e}")
        return None

def train_single_model():
    """Train a single model and return the trained agent."""
    print("\n🚀 Training single model...")
    
    # Setup with fixed seed for training
    training_seed = 42
    key = jax.random.PRNGKey(training_seed)
    
    # Parameters
    num_steps = 30001
    batch_size = 1000
    obs_dim = 10
    action_dim = 2
    
    # Create config and agent
    config = create_config()
    agent = create_agent(config, obs_dim, action_dim, training_seed)
    
    # Training loop
    print("🏋️ Starting training...")
    for step in tqdm(range(num_steps), desc="Training"):
        key, data_key = jax.random.split(key)
        actions = generate_toy_data(batch_size, data_key)
        observations = create_observations(batch_size, obs_dim)
        
        batch = {'observations': observations, 'actions': actions}
        
        # Train with flow-only loss
        new_rng, rng = jax.random.split(agent.rng)
        
        def loss_fn(grad_params):
            return flow_only_loss(agent, batch, grad_params, rng)
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent = agent.replace(network=new_network, rng=new_rng)
        
        if step % 2000 == 0:
            flow_loss = float(info['flow_loss'])
            print(f"Training Step {step:4d}: Loss = {flow_loss:.6f}")
    
    print("✅ Training completed!")
    return agent

def run_multiple_inference_tests(agent, num_tests=5):
    """Run multiple inference tests on the trained model."""
    print(f"\n🧪 Running {num_tests} inference tests...")
    
    # Generate true distribution samples (fixed seed for consistency)
    true_key = jax.random.PRNGKey(123)
    true_samples = np.array(generate_toy_data(5000, true_key))
    
    # Parameters for inference
    obs_dim = 10
    test_obs = create_observations(2000, obs_dim)
    
    results = []
    for i in range(num_tests):
        print(f"🔍 Inference test {i + 1}/{num_tests}")
        
        # Use different random seed for each inference test
        test_seed = 1000 + i * 100
        test_key = jax.random.PRNGKey(test_seed)
        
        # Sample actions from the trained model
        learned_samples = np.array(agent.sample_actions(test_obs, seed=test_key))
        
        # Compute 2D Wasserstein distance
        wasserstein_dist = compute_2d_wasserstein_distance(learned_samples, true_samples)
        
        if wasserstein_dist is not None:
            results.append(wasserstein_dist)
            print(f"   Test {i + 1}: 2D Wasserstein = {wasserstein_dist:.6f}")
        else:
            print(f"   Test {i + 1}: Failed to compute 2D Wasserstein")
    
    return results

def compute_inference_statistics(results):
    """Compute statistics from inference test results."""
    if len(results) == 0:
        print("❌ All inference tests failed")
        return None
    
    # Compute statistics
    results = np.array(results)
    mean_result = np.mean(results)
    std_result = np.std(results, ddof=1) if len(results) > 1 else 0.0
    min_result = np.min(results)
    max_result = np.max(results)
    
    # 95% confidence interval (if we have enough samples)
    if len(results) > 1:
        from scipy import stats
        confidence_interval = stats.t.interval(0.95, len(results)-1, 
                                             loc=mean_result, 
                                             scale=stats.sem(results))
    else:
        confidence_interval = (mean_result, mean_result)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 Inference Test Statistics ({len(results)}/5 successful)")
    print(f"{'='*80}")
    print(f"🎯 2D Wasserstein Distance Statistics:")
    print(f"   Mean:               {mean_result:.6f}")
    print(f"   Std Dev:            {std_result:.6f}")
    print(f"   Min:                {min_result:.6f}")
    print(f"   Max:                {max_result:.6f}")
    print(f"   95% CI:             [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
    if mean_result > 0:
        print(f"   Coefficient of Var: {(std_result/mean_result)*100:.2f}%")
    
    print(f"\n📋 Individual Test Results:")
    for i, result in enumerate(results):
        print(f"   Test {i+1}: {result:.6f}")
    
    # Evaluation
    print(f"\n🔍 Performance Evaluation:")
    if mean_result < 0.1:
        print(f"   ✅ Excellent: Mean 2D Wasserstein < 0.1")
    elif mean_result < 0.2:
        print(f"   ✅ Good: Mean 2D Wasserstein < 0.2")
    elif mean_result < 0.3:
        print(f"   ⚠️  Fair: Mean 2D Wasserstein < 0.3")
    else:
        print(f"   ❌ Poor: Mean 2D Wasserstein >= 0.3")
    
    if std_result < 0.05:
        print(f"   ✅ Very stable inference: Std Dev < 0.05")
    elif std_result < 0.1:
        print(f"   ✅ Stable inference: Std Dev < 0.1")
    else:
        print(f"   ⚠️  Unstable inference: Std Dev >= 0.1")
    
    return {
        'mean': mean_result,
        'std': std_result,
        'min': min_result,
        'max': max_result,
        'confidence_interval': confidence_interval,
        'results': results,
        'success_rate': len(results) / 5
    }

def main():
    """Main function: Train once, test multiple times."""
    print("Testing FQL fitting capability - Single training, multiple inference tests...")
    
    # Setup GPU
    setup_gpu()
    
    # Create visualization directory
    vis_demo_dir = 'vis_demo'
    if not os.path.exists(vis_demo_dir):
        os.makedirs(vis_demo_dir)
    
    # Train single model
    trained_agent = train_single_model()
    
    # Run multiple inference tests
    inference_results = run_multiple_inference_tests(trained_agent, num_tests=5)
    
    # Compute and display statistics
    stats_results = compute_inference_statistics(inference_results)
    
    if stats_results:
        print(f"\n🎉 Single training + multiple inference analysis completed!")
        print(f"📁 Results saved in: {vis_demo_dir}/")
    else:
        print(f"\n❌ Analysis failed")

if __name__ == '__main__':
    main()