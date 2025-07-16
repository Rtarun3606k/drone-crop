"use client";

import React, { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const AleartBox = ({
  message,
  type = "success",
  isVisible = false,
  onClose,
  duration = 5000,
  showCloseButton = true,
}) => {
  useEffect(() => {
    if (isVisible && duration > 0) {
      const timer = setTimeout(() => {
        onClose && onClose();
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [isVisible, duration, onClose]);

  const getTypeConfig = () => {
    switch (type) {
      case "error":
        return {
          bg: "bg-red-900/80 border-red-500",
          text: "text-red-100",
          icon: (
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          ),
          iconBg: "bg-red-500/20",
          progressBar: "bg-red-500",
          title: "Error",
        };
      case "warning":
        return {
          bg: "bg-yellow-900/80 border-yellow-500",
          text: "text-yellow-100",
          icon: (
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
          ),
          iconBg: "bg-yellow-500/20",
          progressBar: "bg-yellow-500",
          title: "Warning",
        };
      case "success":
      default:
        return {
          bg: "bg-green-900/80 border-green-500",
          text: "text-green-100",
          icon: (
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          ),
          iconBg: "bg-green-500/20",
          progressBar: "bg-green-500",
          title: "Success",
        };
    }
  };

  const config = getTypeConfig();

  const backdropVariants = {
    hidden: {
      opacity: 0,
    },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.3,
      },
    },
    exit: {
      opacity: 0,
      transition: {
        duration: 0.3,
      },
    },
  };

  const alertVariants = {
    hidden: {
      opacity: 0,
      scale: 0.7,
      y: -50,
    },
    visible: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: {
        type: "spring",
        damping: 25,
        stiffness: 500,
        duration: 0.5,
      },
    },
    exit: {
      opacity: 0,
      scale: 0.8,
      y: -30,
      transition: {
        duration: 0.3,
      },
    },
  };

  const progressVariants = {
    hidden: { width: "100%" },
    visible: {
      width: "0%",
      transition: {
        duration: duration / 1000,
        ease: "linear",
      },
    },
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          variants={backdropVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
        >
          {/* Background Blur Overlay */}
          <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

          {/* Alert Box */}
          <motion.div
            className={`
              relative z-10
              ${config.bg} ${config.text}
              border-2 
              rounded-xl shadow-2xl
              max-w-md w-full
              overflow-hidden
            `}
            variants={alertVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
          >
            {/* Header */}
            <div className="p-6 pb-4">
              <div className="flex items-start gap-4">
                {/* Icon */}
                <motion.div
                  className={`
                    ${config.iconBg} 
                    rounded-full p-3 flex-shrink-0
                  `}
                  initial={{ rotate: 0, scale: 0 }}
                  animate={{ rotate: 360, scale: 1 }}
                  transition={{
                    delay: 0.2,
                    type: "spring",
                    stiffness: 200,
                  }}
                >
                  {config.icon}
                </motion.div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <motion.h3
                    className="text-lg font-semibold mb-2"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    {config.title}
                  </motion.h3>
                  <motion.p
                    className="text-sm leading-relaxed"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    {message}
                  </motion.p>
                </div>

                {/* Close Button */}
                {showCloseButton && (
                  <motion.button
                    onClick={onClose}
                    className={`
                      ${config.text} hover:opacity-70
                      transition-opacity duration-200
                      flex-shrink-0 p-2 rounded-lg
                      focus:outline-none focus:ring-2 focus:ring-white/20
                      hover:bg-white/10
                    `}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    initial={{ opacity: 0, rotate: -90 }}
                    animate={{ opacity: 1, rotate: 0 }}
                    transition={{ delay: 0.5 }}
                  >
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </motion.button>
                )}
              </div>
            </div>

            {/* Progress Bar */}
            {duration > 0 && (
              <div className="h-1 bg-black/20 relative overflow-hidden">
                <motion.div
                  className={`absolute left-0 top-0 h-full ${config.progressBar}`}
                  variants={progressVariants}
                  initial="hidden"
                  animate="visible"
                />
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// Hook for managing alert state (renamed to avoid collision with popup)
export const useAlert = () => {
  const [alertData, setAlertData] = React.useState(null);

  const displayAlert = (message, type = "success", options = {}) => {
    setAlertData({
      message,
      type,
      isVisible: true,
      id: Date.now(),
      ...options,
    });
  };

  const closeAlert = () => {
    setAlertData((prev) => (prev ? { ...prev, isVisible: false } : null));
  };

  const alertSuccess = (message, options = {}) =>
    displayAlert(message, "success", options);
  const alertError = (message, options = {}) =>
    displayAlert(message, "error", options);
  const alertWarning = (message, options = {}) =>
    displayAlert(message, "warning", options);

  return {
    alertData,
    displayAlert,
    closeAlert,
    alertSuccess,
    alertError,
    alertWarning,
  };
};

export default AleartBox;
