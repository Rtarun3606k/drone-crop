import React, { useState, useEffect } from "react";
import { FiHome, FiMapPin, FiInfo } from "react-icons/fi";
import { useTranslations } from "next-intl";

const HomeLocationInfo = ({ onAlert, refreshTrigger, onUseAsDefault }) => {
  const t = useTranslations("homeLocationInfo");
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
              <strong>{t("alerts.defaultApplied.title")}</strong>
            </p>
            <p>
              {t("alerts.defaultApplied.description")}
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
            {t("title")}
          </h4>

          {homeLocation ? (
            <div>
              <p className="text-gray-300 text-sm mb-2">
                {t("hasLocation.description")}
              </p>
              <div className="bg-green-900/20 border border-green-700 rounded p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center text-green-400 text-sm">
                    <FiHome className="mr-2" size={14} />
                    <strong>{t("hasLocation.currentLocationLabel")}</strong>
                  </div>
                  <button
                    type="button"
                    onClick={handleUseAsDefault}
                    className="bg-green-600 hover:bg-green-700 text-white text-xs px-3 py-1 rounded transition-colors"
                  >
                    {t("hasLocation.useForUploadButton")}
                  </button>
                </div>
                <p className="text-green-300 text-sm">{homeLocation.address}</p>
                <p className="text-green-400 text-xs mt-1">
                  {t("hasLocation.coordinatesLabel", {
                    lat: homeLocation.lat.toFixed(6),
                    lng: homeLocation.lng.toFixed(6)
                  })}
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  if (onAlert) {
                    onAlert(
                      <div>
                        <p>
                          <strong>{t("alerts.howItWorks.title")}</strong>
                        </p>
                        <ul className="list-disc list-inside mt-2 space-y-1 text-sm">
                          {t.raw("alerts.howItWorks.points").map((point, index) => (
                            <li key={index}>{point}</li>
                          ))}
                        </ul>
                      </div>,
                      "info"
                    );
                  }
                }}
                className="text-blue-400 hover:text-blue-300 text-xs underline mt-2"
              >
                {t("hasLocation.learnMoreButton")}
              </button>
            </div>
          ) : (
            <div>
              <p className="text-yellow-300 text-sm mb-2">
                <strong>{t("noLocation.title")}</strong>
              </p>
              <p className="text-gray-300 text-sm mb-3">
                {t("noLocation.description")}
              </p>
              <div className="bg-yellow-900/20 border border-yellow-700 rounded p-3">
                <p className="text-yellow-200 text-xs">
                  💡 <strong>{t("noLocation.tipTitle")}</strong> {t("noLocation.tipDescription")}
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
