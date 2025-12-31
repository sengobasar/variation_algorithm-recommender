const classify = (problemDescription) => {
    if (problemDescription.toLowerCase().includes('q-learning')) {
        return {
            type: 'Reinforcement',
            algorithm: 'Q-Learning',
            description: 'Model-free reinforcement learning algorithm to learn the value of actions.'
        };
    } else if (problemDescription.toLowerCase().includes('deep q') || 
              problemDescription.toLowerCase().includes('dqn')) {
        return {
            type: 'Reinforcement',
            algorithm: 'Deep Q-Network (DQN)',
            description: 'Combines Q-Learning with deep neural networks.'
        };
    }
    
    return {
        type: 'Reinforcement',
        algorithm: 'Policy Gradient',
        description: 'Directly optimizes the policy using gradient ascent.'
    };
};

const getDetails = () => ({
    name: 'Reinforcement Learning',
    description: 'Algorithms that learn by interacting with an environment to maximize rewards.',
    useCases: [
        'Game playing agents',
        'Robotics control',
        'Autonomous vehicles',
        'Resource management'
    ],
    commonAlgorithms: [
        'Q-Learning',
        'Deep Q-Network (DQN)',
        'Policy Gradients',
        'Actor-Critic',
        'Proximal Policy Optimization (PPO)'
    ]
});

module.exports = {
    classify,
    getDetails
};
