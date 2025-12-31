const classify = (problemDescription) => {
    if (problemDescription.toLowerCase().includes('cluster')) {
        return {
            type: 'Unsupervised',
            subType: 'Clustering',
            algorithm: 'K-Means',
            description: 'Groups similar data points into clusters.'
        };
    } else if (problemDescription.toLowerCase().includes('dimension')) {
        return {
            type: 'Unsupervised',
            subType: 'Dimensionality Reduction',
            algorithm: 'PCA',
            description: 'Reduces the number of random variables under consideration.'
        };
    }
    
    return {
        type: 'Unsupervised',
        subType: 'General',
        algorithm: 'Hierarchical Clustering',
        description: 'Builds a hierarchy of clusters.'
    };
};

const getDetails = () => ({
    name: 'Unsupervised Learning',
    description: 'Algorithms that find patterns in unlabeled data.',
    useCases: [
        'Customer segmentation',
        'Anomaly detection',
        'Recommendation systems',
        'Feature extraction'
    ],
    commonAlgorithms: [
        'K-Means',
        'Hierarchical Clustering',
        'DBSCAN',
        'Principal Component Analysis (PCA)',
        'Apriori'
    ]
});

module.exports = {
    classify,
    getDetails
};
