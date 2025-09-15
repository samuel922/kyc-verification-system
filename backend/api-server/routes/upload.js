const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');
const auth = require('../middleware/auth');
const db = require('../config/database');

const router = express.Router();

// Configure multer for file upload
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024 // 10MB
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  }
});

// Upload reference face
router.post('/face', auth, upload.single('face'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No face image provided' });
    }

    const userId = req.user.id;
    const imageBuffer = req.file.buffer;
    
    // Process image with Sharp (resize, optimize)
    const processedImage = await sharp(imageBuffer)
      .resize(512, 512, { fit: 'cover' })
      .jpeg({ quality: 90 })
      .toBuffer();

    // Generate unique filename
    const filename = `face_${userId}_${uuidv4()}.jpg`;
    const filepath = path.join(process.env.UPLOAD_DIR, filename);
    
    // Save file
    await fs.writeFile(filepath, processedImage);
    
    // Call AI service for face detection
    const aiResponse = await axios.post(`${process.env.AI_SERVICE_URL}/detect-face`, {
      image_data: processedImage.toString('base64')
    });

    if (!aiResponse.data.face_detected) {
      // Delete the saved file if no face detected
      await fs.unlink(filepath);
      return res.status(400).json({ error: 'No face detected in image' });
    }

    // Store in database
    const result = await db.query(
      `INSERT INTO user_documents (user_id, document_type, file_path, file_hash, file_size, mime_type)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`,
      [userId, 'reference_face', filepath, 'hash', processedImage.length, 'image/jpeg']
    );

    res.json({
      success: true,
      imageId: result.rows[0].id,
      faceDetected: true,
      confidence: aiResponse.data.confidence
    });

  } catch (error) {
    console.error('Upload face error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Upload ID document
router.post('/id', auth, upload.single('id'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No ID document provided' });
    }

    const userId = req.user.id;
    const imageBuffer = req.file.buffer;
    
    // Process image
    const processedImage = await sharp(imageBuffer)
      .jpeg({ quality: 95 })
      .toBuffer();

    // Generate unique filename
    const filename = `id_${userId}_${uuidv4()}.jpg`;
    const filepath = path.join(process.env.UPLOAD_DIR, filename);
    
    // Save file
    await fs.writeFile(filepath, processedImage);
    
    // Call AI service for face extraction and OCR
    const aiResponse = await axios.post(`${process.env.AI_SERVICE_URL}/process-id`, {
      image_data: processedImage.toString('base64')
    });

    // Store in database
    const result = await db.query(
      `INSERT INTO user_documents (user_id, document_type, file_path, file_hash, file_size, mime_type)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`,
      [userId, 'id_document', filepath, 'hash', processedImage.length, 'image/jpeg']
    );

    res.json({
      success: true,
      documentId: result.rows[0].id,
      faceDetected: aiResponse.data.face_detected,
      ocrData: aiResponse.data.ocr_data || {},
      confidence: aiResponse.data.confidence
    });

  } catch (error) {
    console.error('Upload ID error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;