"use client";

import React, { useEffect, useState } from "react";

const Popup = ({
  message,
  type = "success",
  isVisible = false,
  onClose,
  duration = 5000,
  position = "top-right",
}) => {
  const [show, setShow] = useState(isVisible);

  useEffect(() => {
    setShow(isVisible);
  }, [isVisible]);

  useEffect(() => {
    if (show && duration > 0) {
      const timer = setTimeout(() => {
        handleClose();
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [show, duration]);

  const handleClose = () => {
    setShow(false);
    setTimeout(() => {
      onClose && onClose();
    }, 300); // Wait for animation to complete
  };

  const getTypeStyles = () => {
    switch (type) {
      case "error":
        return {
          bg: "bg-red-900/90 border-red-500",
          text: "text-red-100",
          icon: "❌",
          iconBg: "bg-red-500/20",
          progress: "bg-red-500",
        };
      case "warning":
        return {
          bg: "bg-yellow-900/90 border-yellow-500",
          text: "text-yellow-100",
          icon: "⚠️",
          iconBg: "bg-yellow-500/20",
          progress: "bg-yellow-500",
        };
      case "success":
      default:
        return {
          bg: "bg-green-900/90 border-green-500",
          text: "text-green-100",
          icon: "✅",
          iconBg: "bg-green-500/20",
          progress: "bg-green-500",
        };
    }
  };

  const getPositionStyles = () => {
    switch (position) {
      case "top-left":
        return "top-4 left-4";
      case "top-center":
        return "top-4 left-1/2 transform -translate-x-1/2";
      case "top-right":
      default:
        return "top-4 right-4";
      case "bottom-left":
        return "bottom-4 left-4";
      case "bottom-center":
        return "bottom-4 left-1/2 transform -translate-x-1/2";
      case "bottom-right":
        return "bottom-4 right-4";
    }
  };

  const styles = getTypeStyles();

  if (!show) return null;

  return (
    <div
      className={`
      fixed z-50 
      ${getPositionStyles()}
      transition-all duration-300 ease-in-out
      ${show ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-2"}
    `}
    >
      <div
        className={`
        ${styles.bg} ${styles.text}
        border-2 backdrop-blur-sm
        rounded-lg shadow-2xl
        min-w-80 max-w-md
        overflow-hidden
        transform transition-all duration-300 ease-out
        ${show ? "scale-100" : "scale-95"}
      `}
      >
        {/* Main Content */}
        <div className="p-4 flex items-start gap-3">
          {/* Icon */}
          <div
            className={`
            ${styles.iconBg} 
            rounded-full p-2 flex-shrink-0
            flex items-center justify-center
          `}
          >
            <span className="text-lg">{styles.icon}</span>
          </div>

          {/* Message */}
          <div className="flex-1 pt-1">
            <p className="text-sm font-medium leading-relaxed">{message}</p>
          </div>

          {/* Close Button */}
          <button
            onClick={handleClose}
            className={`
              ${styles.text} hover:opacity-70
              transition-opacity duration-200
              flex-shrink-0 p-1 rounded
              focus:outline-none focus:ring-2 focus:ring-white/20
            `}
            aria-label="Close notification"
          >
            <svg
              className="w-4 h-4"
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
          </button>
        </div>

        {/* Progress Bar (if duration is set) */}
        {duration > 0 && (
          <div className="h-1 bg-black/20">
            <div
              className={`
                h-full ${styles.progress}
                transition-all ease-linear
                ${show ? "w-0" : "w-full"}
              `}
              style={{
                transitionDuration: show ? `${duration}ms` : "0ms",
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

// Hook for easy popup management
export const usePopup = () => {
  const [popup, setPopup] = useState(null);

  const showPopup = (message, type = "success", options = {}) => {
    setPopup({
      message,
      type,
      isVisible: true,
      id: Date.now(),
      ...options,
    });
  };

  const hidePopup = () => {
    setPopup((prev) => (prev ? { ...prev, isVisible: false } : null));
  };

  const showSuccess = (message, options = {}) =>
    showPopup(message, "success", options);
  const showError = (message, options = {}) =>
    showPopup(message, "error", options);
  const showWarning = (message, options = {}) =>
    showPopup(message, "warning", options);

  return {
    popup,
    showPopup,
    hidePopup,
    showSuccess,
    showError,
    showWarning,
  };
};

// Global Popup Provider Component
export const PopupProvider = ({ children }) => {
  const { popup, hidePopup } = usePopup();

  return (
    <>
      {children}
      {popup && (
        <Popup
          message={popup.message}
          type={popup.type}
          isVisible={popup.isVisible}
          onClose={hidePopup}
          duration={popup.duration}
          position={popup.position}
        />
      )}
    </>
  );
};

export default Popup;
