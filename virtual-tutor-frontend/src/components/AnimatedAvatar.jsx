// src/components/AnimatedAvatar.jsx
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const AnimatedAvatar = ({ 
  avatar, 
  isSpeaking, 
  currentWord = '',
  gender = 'male' 
}) => {
  const [mouthState, setMouthState] = useState('closed');
  const [headRotation, setHeadRotation] = useState(0);
  const [handGesture, setHandGesture] = useState('neutral');
  const [blinkState, setBlinkState] = useState(false);
  
  // Lip sync based on phonemes
  useEffect(() => {
    if (!isSpeaking) {
      setMouthState('closed');
      return;
    }

    // Simple lip sync based on vowels and consonants
    const word = currentWord.toLowerCase();
    
    if (word.match(/[aeiou]/)) {
      // Open mouth for vowels
      if (word.match(/[ao]/)) {
        setMouthState('open-wide');
      } else if (word.match(/[ei]/)) {
        setMouthState('open-narrow');
      } else {
        setMouthState('open-medium');
      }
    } else if (word.match(/[mnp]/)) {
      // Close mouth for m, n, p
      setMouthState('closed');
    } else {
      // Slight open for other consonants
      setMouthState('open-slight');
    }
  }, [currentWord, isSpeaking]);

  // Head movement during speech
  useEffect(() => {
    if (!isSpeaking) {
      setHeadRotation(0);
      return;
    }

    const interval = setInterval(() => {
      // Subtle head movements
      setHeadRotation(Math.random() * 6 - 3); // -3 to 3 degrees
    }, 2000);

    return () => clearInterval(interval);
  }, [isSpeaking]);

  // Hand gestures during speech
  useEffect(() => {
    if (!isSpeaking) {
      setHandGesture('neutral');
      return;
    }

    const gestures = ['neutral', 'pointing', 'explaining', 'thinking'];
    const interval = setInterval(() => {
      setHandGesture(gestures[Math.floor(Math.random() * gestures.length)]);
    }, 3000);

    return () => clearInterval(interval);
  }, [isSpeaking]);

  // Blinking animation
  useEffect(() => {
    const blinkInterval = setInterval(() => {
      setBlinkState(true);
      setTimeout(() => setBlinkState(false), 150);
    }, 3000 + Math.random() * 2000);

    return () => clearInterval(blinkInterval);
  }, []);

  // Mouth shapes for lip sync
  const getMouthPath = () => {
    switch (mouthState) {
      case 'open-wide':
        return 'M 40 50 Q 50 60 60 50'; // Wide O shape
      case 'open-medium':
        return 'M 42 50 Q 50 56 58 50'; // Medium open
      case 'open-narrow':
        return 'M 44 50 Q 50 54 56 50'; // E shape
      case 'open-slight':
        return 'M 45 50 Q 50 52 55 50'; // Slight open
      default:
        return 'M 45 50 L 55 50'; // Closed
    }
  };

  return (
    <motion.div
      className="avatar-container"
      animate={{
        rotate: headRotation,
        y: isSpeaking ? [0, -2, 0] : 0,
      }}
      transition={{
        rotate: { duration: 0.5 },
        y: { duration: 1, repeat: isSpeaking ? Infinity : 0 }
      }}
    >
      {/* Avatar Image */}
      <div className="relative w-full h-full">
        <img
          src={avatar.image}
          alt={avatar.name}
          className="w-full h-full object-contain"
        />

        {/* Overlay for lip sync and gestures */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 100 100"
        >
          {/* Mouth animation overlay */}
          <motion.path
            d={getMouthPath()}
            stroke={gender === 'male' ? '#8B4513' : '#C04848'}
            strokeWidth="2"
            fill="none"
            className="mouth-overlay"
            style={{
              transformOrigin: '50% 50%',
            }}
            animate={{
              opacity: isSpeaking ? [0.6, 0.9, 0.6] : 0.3,
            }}
            transition={{
              duration: 0.3,
              repeat: isSpeaking ? Infinity : 0
            }}
          />

          {/* Eye blink overlay */}
          <AnimatePresence>
            {blinkState && (
              <>
                <motion.ellipse
                  cx="35"
                  cy="30"
                  rx="3"
                  ry="0.5"
                  fill="rgba(0,0,0,0.3)"
                  initial={{ scaleY: 1 }}
                  animate={{ scaleY: 0 }}
                  exit={{ scaleY: 1 }}
                  transition={{ duration: 0.1 }}
                />
                <motion.ellipse
                  cx="65"
                  cy="30"
                  rx="3"
                  ry="0.5"
                  fill="rgba(0,0,0,0.3)"
                  initial={{ scaleY: 1 }}
                  animate={{ scaleY: 0 }}
                  exit={{ scaleY: 1 }}
                  transition={{ duration: 0.1 }}
                />
              </>
            )}
          </AnimatePresence>

          {/* Gesture indicators */}
          {isSpeaking && (
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.4 }}
              exit={{ opacity: 0 }}
            >
              {handGesture === 'pointing' && (
                <motion.path
                  d="M 70 60 L 85 50"
                  stroke="#4A90E2"
                  strokeWidth="2"
                  animate={{ x: [0, 3, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                />
              )}
              {handGesture === 'explaining' && (
                <motion.circle
                  cx="75"
                  cy="55"
                  r="5"
                  stroke="#4A90E2"
                  strokeWidth="1.5"
                  fill="none"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              )}
              {handGesture === 'thinking' && (
                <motion.path
                  d="M 30 45 Q 25 40 30 35"
                  stroke="#4A90E2"
                  strokeWidth="2"
                  fill="none"
                  animate={{ opacity: [0.3, 0.7, 0.3] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                />
              )}
            </motion.g>
          )}
        </svg>

        {/* Speaking indicator */}
        <AnimatePresence>
          {isSpeaking && (
            <motion.div
              className="absolute bottom-4 left-1/2 transform -translate-x-1/2"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <div className="flex space-x-1">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-2 h-2 bg-green-500 rounded-full"
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      delay: i * 0.2,
                    }}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

export default AnimatedAvatar;
