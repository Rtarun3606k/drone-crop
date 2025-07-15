import { useEffect, useRef } from "react";
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

const MapSelect = ({
  setSelectedCoordinatesProp,
  setAddressProp,
  selectedCoordinatesProp,
  addressProp,
}) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const vectorSourceRef = useRef(null);

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

    mapInstanceRef.current = new Map({
      target: mapRef.current,
      layers: [new TileLayer({ source: new OSM() }), vectorLayer],
      view: new View({
        center: fromLonLat([80.329, 23.512]),
        zoom: 4,
      }),
    });

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
  }, []);

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
              Selected Location:
            </h4>
            <div className="mb-2 flex gap-2.5">
              <div>
                <strong>Latitude:</strong>{" "}
                {selectedCoordinatesProp.latitude.toFixed(6)}°
              </div>
              <div>
                <strong>Longitude:</strong>{" "}
                {selectedCoordinatesProp.longitude.toFixed(6)}°
              </div>
            </div>
            {addressProp && (
              <div className="mb-2">
                <strong>Address:</strong> {addressProp}
              </div>
            )}
          </div>
        ) : (
          <div className="text-gray-500 italic">
            Click on the map to select a location
          </div>
        )}
      </div>
    </div>
  );
};

export default MapSelect;
