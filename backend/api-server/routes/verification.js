const express = require('express');
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');
const auth = require('../middleware/auth');
const db = require('../config/database');
const redis = require('../config/redis');

const router = express.Router();

// Start verification session
router.post('/start', auth, async (req, res) => {
  try {
    const userId = req.user.id;
    const sessionId = uuidv4();
    
    // Check if user has uploaded required documents
    const documents = await db.query(
      'SELECT document_type FROM user_documents WHERE user_id = $1',
      [userId]
    );
    
    const docTypes = documents.rows.map(doc => doc.document_type);
    const hasReferenceFace = docTypes.includes('reference_face');
    const hasIDDocument = docTypes.includes('id_document');
    
    if (!hasReferenceFace || !hasIDDocument) {
      return res.status(400).json({ 
        error: 'Missing required documents',
        hasReferenceFace,
        hasIDDocument
      });
    }

    // Create verification session
    const session = {
      sessionId,
      userId,
      status: 'active',
      startTime: new Date().toISOString(),
      frameCount: 0,
      livenessChecks: {
        blinkDetected: false,
        headMovement: false,
        livenessScore: 0.0
      },
      verificationResult: null
    };
    
    // Store in Redis with 5 minute expiry
    await redis.setEx(`session:${sessionId}`, 300, JSON.stringify(session));
    
    // Also store in database
    await db.query(
      `INSERT INTO verification_sessions (id, user_id, expires_at, ip_address)
       VALUES ($1, $2, $3, $4)`,
      [sessionId, userId, new Date(Date.now() + 300000), req.ip]
    );

    res.json({
      success: true,
      sessionId,
      expiresIn: 300
    });

  } catch (error) {
    console.error('Start verification error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Process verification frame
router.post('/frame', auth, async (req, res) => {
  try {
    const { sessionId, frameData } = req.body;
    
    if (!sessionId || !frameData) {
      return res.status(400).json({ error: 'Missing sessionId or frameData' });
    }

    // Get session from Redis
    const sessionData = await redis.get(`session:${sessionId}`);
    if (!sessionData) {
      return res.status(400).json({ error: 'Invalid or expired session' });
    }

    const session = JSON.parse(sessionData);
    
    // Call AI service for frame processing
    const aiResponse = await axios.post(`${process.env.AI_SERVICE_URL}/process-frame`, {
      session_id: sessionId,
      user_id: session.userId,
      frame_data: frameData
    });

    // Update session data
    session.frameCount++;
    session.livenessChecks = aiResponse.data.liveness;
    
    // Check if verification is complete
    if (aiResponse.data.liveness.is_live && aiResponse.data.face_match.confidence > 0.8) {
      session.status = 'completed';
      session.verificationResult = {
        success: true,
        confidence: aiResponse.data.face_match.confidence,
        completedAt: new Date().toISOString()
      };
      
      // Update database
      await db.query(
        `UPDATE verification_sessions 
         SET status = $1, completed_at = $2, verification_result = $3
         WHERE id = $4`,
        ['completed', new Date(), JSON.stringify(session.verificationResult), sessionId]
      );
    }
    
    // Update session in Redis
    await redis.setEx(`session:${sessionId}`, 300, JSON.stringify(session));

    res.json({
      success: true,
      frameProcessed: true,
      liveness: aiResponse.data.liveness,
      faceMatch: aiResponse.data.face_match,
      sessionStatus: session.status,
      maskedFrame: aiResponse.data.masked_frame
    });

  } catch (error) {
    console.error('Process frame error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get verification status
router.get('/:sessionId/status', auth, async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const sessionData = await redis.get(`session:${sessionId}`);
    if (!sessionData) {
      return res.status(404).json({ error: 'Session not found' });
    }

    const session = JSON.parse(sessionData);
    
    res.json({
      sessionId,
      status: session.status,
      frameCount: session.frameCount,
      livenessChecks: session.livenessChecks,
      verificationResult: session.verificationResult
    });

  } catch (error) {
    console.error('Get status error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;