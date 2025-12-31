const classify = (problemDescription) => {
    // Simple keyword matching - can be enhanced with ML model
    if (problemDescription.toLowerCase().includes('regression')) {
        return {
            type: 'Supervised',
            subType: 'Regression',
            algorithm: 'Linear Regression',
            description: 'Predicts continuous values based on input features.'
        };
    } else if (problemDescription.toLowerCase().includes('classification')) {
        return {
            type: 'Supervised',
            subType: 'Classification',
            algorithm: 'Random Forest',
            description: 'Uses multiple decision trees to improve classification accuracy.'
        };
    }
    
    // Default return for supervised learning
    return {
        type: 'Supervised',
        subType: 'General',
        algorithm: 'Gradient Boosting',
        description: 'Ensemble method that builds models sequentially to correct errors.'
    };
};

const getDetails = () => ({
    name: 'Supervised Learning',
    description: 'Algorithms that learn from labeled training data.',
    useCases: [
        'Predicting house prices',
        'Image classification',
        'Spam detection',
        'Customer churn prediction'
    ],
    commonAlgorithms: [
        'Linear/Logistic Regression',
        'Support Vector Machines',
        'Decision Trees',
        'Random Forest',
        'Neural Networks'
    ]
});

module.exports = {
    classify,
    getDetails
};
