const classify = (problemDescription) => {
    if (problemDescription.toLowerCase().includes('image') || 
        problemDescription.toLowerCase().includes('cnn')) {
        return {
            type: 'Deep Learning',
            subType: 'Computer Vision',
            algorithm: 'Convolutional Neural Network (CNN)',
            description: 'Specialized for processing structured grid data like images.'
        };
    } else if (problemDescription.toLowerCase().includes('sequence') || 
              problemDescription.toLowerCase().includes('rnn') ||
              problemDescription.toLowerCase().includes('lstm')) {
        return {
            type: 'Deep Learning',
            subType: 'Sequence Processing',
            algorithm: 'LSTM/GRU',
            description: 'Specialized for sequential data like time series or text.'
        };
    } else if (problemDescription.toLowerCase().includes('transformer') || 
              problemDescription.toLowerCase().includes('attention')) {
        return {
            type: 'Deep Learning',
            subType: 'NLP/Sequence',
            algorithm: 'Transformer',
            description: 'Uses self-attention mechanisms for processing sequences.'
        };
    }
    
    return {
        type: 'Deep Learning',
        subType: 'General',
        algorithm: 'Deep Neural Network',
        description: 'Standard feedforward neural network with multiple hidden layers.'
    };
};

const getDetails = () => ({
    name: 'Deep Learning',
    description: 'Neural networks with multiple layers that can learn representations of data.',
    useCases: [
        'Image recognition',
        'Natural language processing',
        'Speech recognition',
        'Autonomous vehicles'
    ],
    commonAlgorithms: [
        'Convolutional Neural Networks (CNN)',
        'Recurrent Neural Networks (RNN)',
        'Long Short-Term Memory (LSTM)',
        'Transformers',
        'Generative Adversarial Networks (GANs)'
    ]
});

module.exports = {
    classify,
    getDetails
};
