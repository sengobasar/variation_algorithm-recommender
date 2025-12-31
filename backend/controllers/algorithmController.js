const supervisedService = require('../services/supervisedService');
const unsupervisedService = require('../services/unsupervisedService');
const reinforcementService = require('../services/reinforcementService');
const deepLearningService = require('../services/deepLearningService');

const classifyAlgorithm = async (req, res) => {
    try {
        const { problemDescription } = req.body;
        
        // Simple classification logic - this can be enhanced
        let result;
        if (problemDescription.toLowerCase().includes('labeled') || 
            problemDescription.toLowerCase().includes('supervised')) {
            result = await supervisedService.classify(problemDescription);
        } else if (problemDescription.toLowerCase().includes('unlabeled') || 
                  problemDescription.toLowerCase().includes('unsupervised')) {
            result = await unsupervisedService.classify(problemDescription);
        } else if (problemDescription.toLowerCase().includes('reward') || 
                  problemDescription.toLowerCase().includes('reinforcement')) {
            result = await reinforcementService.classify(problemDescription);
        } else if (problemDescription.toLowerCase().includes('neural') || 
                  problemDescription.toLowerCase().includes('deep learning')) {
            result = await deepLearningService.classify(problemDescription);
        } else {
            // Default to supervised learning if no clear match
            result = await supervisedService.classify(problemDescription);
        }
        
        res.json({ success: true, result });
    } catch (error) {
        console.error('Error in classifyAlgorithm:', error);
        res.status(500).json({ success: false, error: 'Failed to classify algorithm' });
    }
};

const getAlgorithmDetails = async (req, res) => {
    try {
        const { algorithmType } = req.params;
        let details;
        
        switch(algorithmType.toLowerCase()) {
            case 'supervised':
                details = supervisedService.getDetails();
                break;
            case 'unsupervised':
                details = unsupervisedService.getDetails();
                break;
            case 'reinforcement':
                details = reinforcementService.getDetails();
                break;
            case 'deeplearning':
                details = deepLearningService.getDetails();
                break;
            default:
                return res.status(404).json({ success: false, error: 'Algorithm type not found' });
        }
        
        res.json({ success: true, details });
    } catch (error) {
        console.error('Error in getAlgorithmDetails:', error);
        res.status(500).json({ success: false, error: 'Failed to get algorithm details' });
    }
};

module.exports = {
    classifyAlgorithm,
    getAlgorithmDetails
};
