import React, { useState } from "react";
import { FiHome } from "react-icons/fi";

const SetAsHome = ({
  setSelectedCoordinatesProp,
  setAddressProp,
  selectedCoordinatesProp,
  addressProp,
}) => {
  const [Loading, setLoading] = useState(false);

  return (
    <div>
      <div className="flex items-center justify-start mt-2 gap-2.5">
        <strong>Set Work Place as :</strong>
        <button
          className="rounded-3xl bg-green-600 text-sm text-white pr-2 pl-2 pt-1 pb-1 hover:bg-green-700 transition-colors duration-200 flex items-center gap-1.5"
          onClick={() => {
            console.log("Setting as home location...");
          }}
          type="button"
        >
          <FiHome />
          <p>Home</p>
        </button>
      </div>
    </div>
  );
};

export default SetAsHome;
