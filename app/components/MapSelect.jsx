import { useEffect, useRef } from 'react';
import Map from 'ol/Map.js';
import View from 'ol/View.js';
import TileLayer from 'ol/layer/Tile.js';
import OSM from 'ol/source/OSM.js';
import { fromLonLat } from 'ol/proj.js';
import { defaults as defaultControls } from 'ol/control.js';

const MapSelect = () => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);

  useEffect(() => {
    if (!mapRef.current) return;

    // Create the map instance
    mapInstanceRef.current = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM(),
        }),
      ],
      view: new View({
        center: fromLonLat([80.3290,23.5120]),
        zoom: 4,
      }),
      controls: defaultControls({
        attribution: false
      })
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
    <div 
        ref={mapRef} 
        style={{ 
            width: '100%', 
            height: '400px', 
            borderRadius: '4px', 
            overflow: 'hidden', // Add this to ensure the map respects the border radius
            marginTop: '10px',
            color: 'black'
        }}
        className="map-container"
    />
);
};

export default MapSelect;