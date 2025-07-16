import React, { useState, useEffect } from "react";
import { FiHome, FiMapPin, FiInfo } from "react-icons/fi";

const HomeLocationInfo = ({ onAlert, refreshTrigger, onUseAsDefault }) => {
  const [homeLocation, setHomeLocation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [hasLoaded, setHasLoaded] = useState(false);

  const loadHomeLocation = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/user/set-home");
      if (response.ok) {
        const data = await response.json();
        if (data.coordinates) {
          setHomeLocation(data.coordinates);
        } else {
          setHomeLocation(null);
        }
      }
    } catch (error) {
      console.error("Error loading home location:", error);
    } finally {
      setLoading(false);
      setHasLoaded(true);
    }
  };

  useEffect(() => {
    // Only load on mount or when explicitly refreshed
    if (!hasLoaded || refreshTrigger > 0) {
      loadHomeLocation();
    }
  }, [refreshTrigger]); // Refresh when refreshTrigger changes

  const handleUseAsDefault = () => {
    if (homeLocation && onUseAsDefault) {
      onUseAsDefault(homeLocation);
      if (onAlert) {
        onAlert(
          <div>
            <p>
              <strong>Default location applied!</strong>
            </p>
            <p>
              Your existing home location will be used for images without GPS
              data in this upload.
            </p>
          </div>,
          "success"
        );
      }
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-3/4 mb-2"></div>
        <div className="h-3 bg-gray-700 rounded w-1/2"></div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      <div className="flex items-start space-x-3">
        <FiInfo className="text-blue-400 mt-0.5 flex-shrink-0" size={16} />
        <div className="flex-1">
          <h4 className="text-white font-semibold mb-2 flex items-center">
            <FiMapPin className="mr-2" size={16} />
            Default Location for Non-GPS Images
          </h4>

          {homeLocation ? (
            <div>
              <p className="text-gray-300 text-sm mb-2">
                Images without GPS data will use your default workplace
                location:
              </p>
              <div className="bg-green-900/20 border border-green-700 rounded p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center text-green-400 text-sm">
                    <FiHome className="mr-2" size={14} />
                    <strong>Current Default Location</strong>
                  </div>
                  <button
                    type="button"
                    onClick={handleUseAsDefault}
                    className="bg-green-600 hover:bg-green-700 text-white text-xs px-3 py-1 rounded transition-colors"
                  >
                    Use for This Upload
                  </button>
                </div>
                <p className="text-green-300 text-sm">{homeLocation.address}</p>
                <p className="text-green-400 text-xs mt-1">
                  Coordinates: {homeLocation.lat.toFixed(6)},{" "}
                  {homeLocation.lng.toFixed(6)}
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  if (onAlert) {
                    onAlert(
                      <div>
                        <p>
                          <strong>How Default Location Works</strong>
                        </p>
                        <ul className="list-disc list-inside mt-2 space-y-1 text-sm">
                          <li>
                            Images with GPS data will use their original
                            location
                          </li>
                          <li>
                            Images without GPS data will be tagged with your
                            default location
                          </li>
                          <li>
                            You can update your default location anytime using
                            the map below
                          </li>
                          <li>
                            This ensures all images have location data for
                            analysis
                          </li>
                        </ul>
                      </div>,
                      "info"
                    );
                  }
                }}
                className="text-blue-400 hover:text-blue-300 text-xs underline mt-2"
              >
                Learn more about default locations
              </button>
            </div>
          ) : (
            <div>
              <p className="text-yellow-300 text-sm mb-2">
                <strong>No default location set!</strong>
              </p>
              <p className="text-gray-300 text-sm mb-3">
                Images without GPS data won't have location information. Set a
                default workplace location below to ensure proper analysis.
              </p>
              <div className="bg-yellow-900/20 border border-yellow-700 rounded p-3">
                <p className="text-yellow-200 text-xs">
                  💡 <strong>Tip:</strong> Set your primary workplace or field
                  location as default. This will be used for images that don't
                  contain GPS data.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HomeLocationInfo;
