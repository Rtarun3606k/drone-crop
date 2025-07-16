import React, { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { FiHome, FiLoader } from "react-icons/fi";

const SetAsHome = ({
  setSelectedCoordinatesProp,
  setAddressProp,
  selectedCoordinatesProp,
  addressProp,
  onAlert, // Alert function passed from parent
}) => {
  const t = useTranslations("setAsHome");
  const [isLoading, setIsLoading] = useState(false);
  const [existingHome, setExistingHome] = useState(null);

  // Load existing home location on mount
  useEffect(() => {
    const loadExistingHome = async () => {
      try {
        const response = await fetch("/api/user/set-home");
        if (response.ok) {
          const data = await response.json();
          if (data.coordinates) {
            setExistingHome(data.coordinates);
          }
        }
      } catch (error) {
        console.error("Error loading existing home location:", error);
      }
    };

    loadExistingHome();
  }, []);

  const setHomeLocation = async () => {
    // Validate required data
    if (!selectedCoordinatesProp || !addressProp) {
      onAlert &&
        onAlert(t("alerts.selectLocationFirst"), "warning");
      return;
    }

    // Validate coordinate values
    if (
      !selectedCoordinatesProp.latitude ||
      !selectedCoordinatesProp.longitude
    ) {
      onAlert &&
        onAlert(
          t("alerts.invalidCoordinates"),
          "error"
        );
      return;
    }

    setIsLoading(true);

    try {
      const homeData = {
        lat: selectedCoordinatesProp.latitude,
        lng: selectedCoordinatesProp.longitude,
        address: addressProp,
      };

      console.log("Sending home location data:", homeData); // Debug log

      const response = await fetch("/api/user/set-home", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(homeData),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        const isUpdate = existingHome ? true : false;
        setExistingHome(homeData); // Update local state

        onAlert &&
          onAlert(
            <div>
              <p className="font-bold">
                {isUpdate
                  ? t("alerts.homeLocationUpdated")
                  : t("alerts.homeLocationSet")}
              </p>
              <p className="mt-1 text-sm">{t("alerts.workplaceSaved")}</p>
              <p className="mt-1 text-xs bg-green-800/30 rounded p-2">
                {addressProp}
              </p>
              <p className="mt-2 text-xs">
                {t("alerts.coordinates", {
                  lat: selectedCoordinatesProp.latitude.toFixed(6),
                  lng: selectedCoordinatesProp.longitude.toFixed(6)
                })}
              </p>
              <p className="mt-2 text-xs opacity-80">
                {t("alerts.defaultLocationNote")}
              </p>
            </div>,
            "success",
            { duration: 6000 }
          );
      } else {
        throw new Error(data.error || "Failed to set home location");
      }
    } catch (error) {
      console.error("Error setting home location:", error);
      onAlert &&
        onAlert(
          <div>
            <p className="font-bold">{t("alerts.setHomeFailed")}</p>
            <p className="mt-1 text-sm">
              {error.message || t("alerts.tryAgain")}
            </p>
            <p className="mt-2 text-xs">
              {t("alerts.checkConnection")}
            </p>
          </div>,
          "error",
          { duration: 5000 }
        );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-start mt-2 gap-2.5">
        <strong>{t("setWorkplaceAs")}</strong>
        <button
          type="button"
          disabled={isLoading || !selectedCoordinatesProp || !addressProp}
          className={`
            rounded-3xl text-sm text-white pr-3 pl-3 pt-2 pb-2 
            transition-all duration-200 flex items-center gap-2
            ${
              isLoading || !selectedCoordinatesProp || !addressProp
                ? "bg-gray-600 cursor-not-allowed opacity-50"
                : "bg-green-600 hover:bg-green-700 hover:scale-105"
            }
          `}
          onClick={setHomeLocation}
          title={
            !selectedCoordinatesProp || !addressProp
              ? t("tooltips.selectLocation")
              : t("tooltips.setAsHome")
          }
        >
          {isLoading ? <FiLoader className="animate-spin" /> : <FiHome />}
          <span className="font-medium">
            {isLoading ? t("setting") : t("setAsHome")}
          </span>
        </button>
      </div>

      {/* Status indicator */}
      {selectedCoordinatesProp && addressProp && !isLoading && (
        <div className="mt-2 text-xs text-gray-400 flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>{t("locationReady")}</span>
        </div>
      )}
    </div>
  );
};

export default SetAsHome;
