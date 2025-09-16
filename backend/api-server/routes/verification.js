const express = require('express');
const axios = require('axios');

const router = express.Router();
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// Remove auth middleware - this is a demo face swap app

// Upload reference face for swapping
router.post('/upload-reference', async (req, res) => {
  try {
    const { image_data } = req.body;

    if (!image_data) {
      return res.status(400).json({ error: 'Missing image_data' });
    }

    // Forward request to AI service
    const aiResponse = await axios.post(`${AI_SERVICE_URL}/upload-reference-face`, {
      image_data
    });

    res.json(aiResponse.data);

  } catch (error) {
    console.error('Upload reference error:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});

// Process video frame for face swapping
router.post('/process-frame', async (req, res) => {
  try {
    const { frame_data } = req.body;

    if (!frame_data) {
      return res.status(400).json({ error: 'Missing frame_data' });
    }

    // Forward request to AI service
    const aiResponse = await axios.post(`${AI_SERVICE_URL}/process-frame`, {
      frame_data
    });

    res.json(aiResponse.data);

  } catch (error) {
    console.error('Process frame error:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});

// Clear reference face
router.delete('/clear-reference', async (req, res) => {
  try {
    const aiResponse = await axios.delete(`${AI_SERVICE_URL}/clear-reference`);
    res.json(aiResponse.data);

  } catch (error) {
    console.error('Clear reference error:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});

// Get reference face status
router.get('/reference-status', async (req, res) => {
  try {
    const aiResponse = await axios.get(`${AI_SERVICE_URL}/reference-status`);
    res.json(aiResponse.data);

  } catch (error) {
    console.error('Reference status error:', error);
    if (error.response) {
      res.status(error.response.status).json(error.response.data);
    } else {
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});

module.exports = router;