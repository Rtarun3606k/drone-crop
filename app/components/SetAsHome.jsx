import React, { useState, useEffect } from "react";
import { FiHome, FiLoader } from "react-icons/fi";

const SetAsHome = ({
  setSelectedCoordinatesProp,
  setAddressProp,
  selectedCoordinatesProp,
  addressProp,
  onAlert, // Alert function passed from parent
}) => {
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
        onAlert("Please select a location on the map first.", "warning");
      return;
    }

    // Validate coordinate values
    if (
      !selectedCoordinatesProp.latitude ||
      !selectedCoordinatesProp.longitude
    ) {
      onAlert &&
        onAlert(
          "Invalid coordinate data. Please select a location on the map.",
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
                  ? "Home Location Updated! 🏠"
                  : "Home Location Set Successfully! 🏠"}
              </p>
              <p className="mt-1 text-sm">Your workplace has been saved as:</p>
              <p className="mt-1 text-xs bg-green-800/30 rounded p-2">
                {addressProp}
              </p>
              <p className="mt-2 text-xs">
                Coordinates: {selectedCoordinatesProp.latitude.toFixed(6)}°,{" "}
                {selectedCoordinatesProp.longitude.toFixed(6)}°
              </p>
              <p className="mt-2 text-xs opacity-80">
                This location will be used as default for images without GPS
                data.
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
            <p className="font-bold">Failed to Set Home Location</p>
            <p className="mt-1 text-sm">
              {error.message || "Please try again."}
            </p>
            <p className="mt-2 text-xs">
              Make sure you have a stable internet connection and try again.
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
        <strong>Set Work Place as:</strong>
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
              ? "Please select a location on the map first"
              : "Set this location as your home workplace"
          }
        >
          {isLoading ? <FiLoader className="animate-spin" /> : <FiHome />}
          <span className="font-medium">
            {isLoading ? "Setting..." : "Set as Home"}
          </span>
        </button>
      </div>

      {/* Status indicator */}
      {selectedCoordinatesProp && addressProp && !isLoading && (
        <div className="mt-2 text-xs text-gray-400 flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>Location ready to be set as home</span>
        </div>
      )}
    </div>
  );
};

export default SetAsHome;
