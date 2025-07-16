import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import Map from "ol/Map.js";
import View from "ol/View.js";
import TileLayer from "ol/layer/Tile.js";
import VectorLayer from "ol/layer/Vector.js";
import OSM from "ol/source/OSM.js";
import VectorSource from "ol/source/Vector.js";
import Feature from "ol/Feature.js";
import Point from "ol/geom/Point.js";
import Style from "ol/style/Style.js";
import Circle from "ol/style/Circle.js";
import Fill from "ol/style/Fill.js";
import Stroke from "ol/style/Stroke.js";
import { fromLonLat, toLonLat } from "ol/proj.js";
import "ol/ol.css";
import SetAsHome from "./SetAsHome";

const MapSelect = ({
  setSelectedCoordinatesProp,
  setAddressProp,
  selectedCoordinatesProp,
  addressProp,
  onAlert, // Pass through the alert function
  skipHomeLoad = false, // Skip loading home location if already handled by parent
}) => {
  const t = useTranslations("mapSelect");
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const vectorSourceRef = useRef(null);
  const [existingHome, setExistingHome] = useState(null);
  const [hasLoadedHome, setHasLoadedHome] = useState(false);
  const loadedRef = useRef(false);

  // Load existing home location on mount
  useEffect(() => {
    if (skipHomeLoad) {
      setHasLoadedHome(true);
      return;
    }

    if (loadedRef.current) return; // Prevent multiple loads using ref
    loadedRef.current = true;

    const loadExistingHome = async () => {
      try {
        const response = await fetch("/api/user/set-home");
        if (response.ok) {
          const data = await response.json();
          if (data.coordinates) {
            setExistingHome(data.coordinates);

            // Show alert about existing home location (only once)
            if (onAlert && !hasLoadedHome) {
              onAlert(
                <div>
                  <p>
                    <strong>{t("alerts.existingHomeFound.title")}</strong>
                  </p>
                  <p>
                    {t("alerts.existingHomeFound.addressLabel")}{" "}
                    {data.coordinates.address ||
                      `${data.coordinates.lat}, ${data.coordinates.lng}`}
                  </p>
                  <p>
                    {t("alerts.existingHomeFound.updateInfo")}
                  </p>
                </div>,
                "info"
              );
            }

            // If no coordinates are set yet, use existing home as default
            if (
              !selectedCoordinatesProp ||
              (!selectedCoordinatesProp.lat && !selectedCoordinatesProp.lng)
            ) {
              if (setSelectedCoordinatesProp) {
                setSelectedCoordinatesProp({
                  latitude: data.coordinates.lat,
                  longitude: data.coordinates.lng,
                });
              }
              if (setAddressProp) {
                setAddressProp(
                  data.coordinates.address ||
                    `${data.coordinates.lat}, ${data.coordinates.lng}`
                );
              }
            }
          } else {
            // No existing home location (only show alert once)
            if (onAlert && !hasLoadedHome) {
              onAlert(
                <div>
                  <p>
                    <strong>{t("alerts.noHomeLocation.title")}</strong>
                  </p>
                  <p>
                    {t("alerts.noHomeLocation.description")}
                  </p>
                </div>,
                "info"
              );
            }
          }
        }
      } catch (error) {
        console.error("Error loading existing home location:", error);
        if (onAlert && !hasLoadedHome) {
          onAlert(t("alerts.errorLoading"), "error");
        }
      } finally {
        setHasLoadedHome(true);
      }
    };

    loadExistingHome();
  }, [skipHomeLoad]); // Only depend on skipHomeLoad

  const fetchAddressFromCoords = async (lat, lon) => {
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`
      );
      const data = await response.json();
      setAddressProp(data.display_name || "No address found");
    } catch (error) {
      console.error("Error fetching address:", error);
      setAddressProp("Error fetching address");
    }
  };

  useEffect(() => {
    if (!mapRef.current) return;

    vectorSourceRef.current = new VectorSource();

    const vectorLayer = new VectorLayer({
      source: vectorSourceRef.current,
      style: new Style({
        image: new Circle({
          radius: 8,
          fill: new Fill({ color: "#ff0000" }),
          stroke: new Stroke({ color: "#ffffff", width: 2 }),
        }),
      }),
    });

    // Determine initial center and zoom based on existing home or default
    const initialCenter = existingHome
      ? fromLonLat([existingHome.lng, existingHome.lat])
      : fromLonLat([80.329, 23.512]);
    const initialZoom = existingHome ? 12 : 4;

    mapInstanceRef.current = new Map({
      target: mapRef.current,
      layers: [new TileLayer({ source: new OSM() }), vectorLayer],
      view: new View({
        center: initialCenter,
        zoom: initialZoom,
      }),
    });

    // Add existing home marker if available
    if (existingHome) {
      const homeMarker = new Feature({
        geometry: new Point(fromLonLat([existingHome.lng, existingHome.lat])),
      });
      vectorSourceRef.current.addFeature(homeMarker);
    }

    mapInstanceRef.current.on("click", function (event) {
      const coordinate = event.coordinate;
      const lonLat = toLonLat(coordinate);

      const coordinateData = {
        longitude: lonLat[0],
        latitude: lonLat[1],
        projected: coordinate,
      };

      setSelectedCoordinatesProp(coordinateData);
      fetchAddressFromCoords(coordinateData.latitude, coordinateData.longitude);

      vectorSourceRef.current.clear();
      const marker = new Feature({ geometry: new Point(coordinate) });
      vectorSourceRef.current.addFeature(marker);
    });

    return () => {
      mapInstanceRef.current?.setTarget(null);
      mapInstanceRef.current = null;
    };
  }, [existingHome]); // Add existingHome as dependency

  // Update map when existing home is loaded
  useEffect(() => {
    if (mapInstanceRef.current && existingHome) {
      const view = mapInstanceRef.current.getView();
      view.setCenter(fromLonLat([existingHome.lng, existingHome.lat]));
      view.setZoom(12);

      // Add marker for existing home
      if (vectorSourceRef.current) {
        vectorSourceRef.current.clear();
        const homeMarker = new Feature({
          geometry: new Point(fromLonLat([existingHome.lng, existingHome.lat])),
        });
        vectorSourceRef.current.addFeature(homeMarker);
      }
    }
  }, [existingHome]);

  return (
    <div>
      <div
        ref={mapRef}
        className="w-full h-96 rounded-lg mt-2.5 overflow-hidden text-black map-container"
      />
      <div className="bg-gray-900 mt-2.5 border border-gray-700 rounded-lg p-2">
        {selectedCoordinatesProp ? (
          <div className="text-gray-300">
            <h4 className="m-0 mb-2.5 font-sans font-semibold">
              {t("selectedLocation")}
            </h4>
            <div className="mb-2 flex gap-2.5">
              <div>
                <strong>{t("latitude")}</strong>{" "}
                {selectedCoordinatesProp?.latitude?.toFixed(6)}°
              </div>
              <div>
                <strong>{t("longitude")}</strong>{" "}
                {selectedCoordinatesProp?.longitude?.toFixed(6)}°
              </div>
            </div>
            {addressProp && (
              <div className="mb-2">
                <strong>{t("address")}</strong> {addressProp}
              </div>
            )}
            <SetAsHome
              setSelectedCoordinatesProp={setSelectedCoordinatesProp}
              setAddressProp={setAddressProp}
              selectedCoordinatesProp={selectedCoordinatesProp}
              addressProp={addressProp}
              onAlert={onAlert}
            />
          </div>
        ) : (
          <div className="text-gray-500 italic">
            {t("clickToSelect")}
          </div>
        )}
      </div>
    </div>
  );
};

export default MapSelect;
