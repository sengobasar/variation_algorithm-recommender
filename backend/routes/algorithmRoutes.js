const express = require('express');
const router = express.Router();
const algorithmController = require('../controllers/algorithmController');

// Route for algorithm classification
router.post('/classify', algorithmController.classifyAlgorithm);

// Route for getting algorithm details
router.get('/:algorithmType', algorithmController.getAlgorithmDetails);

module.exports = router;
