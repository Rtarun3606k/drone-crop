import { useEffect, useRef, useState } from "react";
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

const MapSelect = () => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const vectorSourceRef = useRef(null);
  const [selectedCoordinates, setSelectedCoordinates] = useState(null);

  useEffect(() => {
    if (!mapRef.current) return;

    // Create a vector source to hold the marker
    vectorSourceRef.current = new VectorSource();

    // Create the vector layer for markers
    const vectorLayer = new VectorLayer({
      source: vectorSourceRef.current,
      style: new Style({
        image: new Circle({
          radius: 8,
          fill: new Fill({
            color: "#ff0000",
          }),
          stroke: new Stroke({
            color: "#ffffff",
            width: 2,
          }),
        }),
      }),
    });

    // Create the map instance
    mapInstanceRef.current = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM(),
        }),
        vectorLayer,
      ],
      view: new View({
        center: fromLonLat([80.329, 23.512]),
        zoom: 4,
      }),
      // controls: defaultControls({
      //   attribution: false
      // })
    });

    // Add click event listener
    mapInstanceRef.current.on("click", function (event) {
      // Get the clicked coordinate
      const coordinate = event.coordinate;

      // Convert to longitude/latitude
      const lonLat = toLonLat(coordinate);

      // Update state with coordinates
      const coordinateData = {
        longitude: lonLat[0],
        latitude: lonLat[1],
        projected: coordinate,
      };

      setSelectedCoordinates(coordinateData);

      // Log the coordinates
      console.log("Selected coordinates:", coordinateData);

      // Clear existing markers
      vectorSourceRef.current.clear();

      // Create a new point feature at the clicked location
      const marker = new Feature({
        geometry: new Point(coordinate),
      });

      // Add the marker to the vector source
      vectorSourceRef.current.addFeature(marker);
    });

    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.setTarget(null);
        mapInstanceRef.current = null;
      }
    };
  }, []);

  return (
    <div>
      <div
        ref={mapRef}
        className="w-full h-96 rounded-lg mt-2.5 overflow-hidden text-black map-container"
      />
      <div className="bg-gray-900 mt-2.5 border border-gray-700 rounded-lg p-2">
        {selectedCoordinates ? (
          <div className="text-gray-300">
            <h4 className="m-0 mb-2.5  font-sans font-semibold">
              Selected Location:
            </h4>
            <div className="mb-2 ">
              <strong>Latitude:</strong> {selectedCoordinates.latitude.toFixed(6)}°
            </div>
            <div className="mb-2 ">
              <strong>Longitude:</strong> {selectedCoordinates.longitude.toFixed(6)}°
            </div>
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