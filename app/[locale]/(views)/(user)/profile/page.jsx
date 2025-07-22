"use client";

import React, { useState } from "react";
import { useSession } from "next-auth/react";
import SessionManager from "@/app/components/SessionManager";
import { City, State } from "country-state-city";
import ProtectedRoute from "@/app/components/ProtectedRoute";
import LanguageSwitcher from "@/app/components/LanguageSwitcher";

import AleartBox, { useAlert } from "@/app/components/AleartBox";
import { FiClipboard } from "react-icons/fi";

// import { State, City } from "country-state-cities";

export default function ProfilePage() {
  const { data: session } = useSession();
  const [errors, setErrors] = useState({});
  const [cities, setCities] = useState([]);

  const [form, setForm] = useState({
    phone: "",
    address: "",
    state: "",
    city: "",
  });

  return (
    <ProtectedRoute>
      <ProfileContent session={session} />
    </ProtectedRoute>
  );
}

function ProfileContent({ session }) {
  const {
    alertData,
    alertSuccess,
    alertError,
    alertWarning,
    alertInfo,
    closeAlert,
  } = useAlert();
  const [isCopied, setIsCopied] = useState(false);
  if (!session) return null;

  const user = session.user;

  // Get all Indian states using country-state-cities
  const allStates = State.getStatesOfCountry("IN");

  // Form state

  // Handle state change to update cities
  const handleStateChange = (e) => {
    const selectedState = e.target.value;
    setForm({ ...form, state: selectedState, city: "" });
    const stateObj = allStates.find((s) => s.name === selectedState);
    if (stateObj) {
      const stateCitiesList = City.getCitiesOfState("IN", stateObj.isoCode);
      setCities(stateCitiesList.map((city) => city.name));
    } else {
      setCities([]);
    }
  };

  // Validate Indian phone number
  const validatePhone = (phone) => {
    return /^[6-9]\d{9}$/.test(phone);
  };

  // Handle form submit
  const handleSubmit = (e) => {
    e.preventDefault();
    let newErrors = {};
    if (!validatePhone(form.phone)) {
      newErrors.phone = "Enter a valid 10-digit Indian phone number";
    }
    if (!form.address) newErrors.address = "Address is required";
    if (!form.state) newErrors.state = "State is required";
    if (!form.city) newErrors.city = "City is required";
    setErrors(newErrors);
    if (Object.keys(newErrors).length === 0) {
      // Submit form or save data
      alert("Profile updated!");
    }
  };
  const handleCopyClick = () => {
    navigator.clipboard
      .writeText(user.mobileId || "No Mobile ID")
      .then(() => {
        // If the text is successfully copied, we set the state to true
        setIsCopied(true);
        // And after 2 seconds, we set it back to false
        alertSuccess(user.mobileId + " Mobile ID copied to clipboard!", {
          duration: 2000,
        });
        setTimeout(() => {
          setIsCopied(false);
        }, 2000);
      })
      .catch((err) => {
        console.error("Failed to copy text: ", err);
        alertError("Failed to copy Mobile ID. Please try again.", {
          duration: 2000,
        });
        setIsCopied(false);
      });
  };

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center py-24">
      <div className="w-full max-w-3xl px-4">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
          Your Profile
        </h1>

        <div className="bg-white dark:bg-gray-900 shadow-lg rounded-lg p-8 mb-8 border border-gray-200 dark:border-gray-800">
          <div className="flex flex-col md:flex-row md:items-center">
            <div className="flex-shrink-0 flex justify-center md:justify-start mb-6 md:mb-0 md:mr-6">
              {user.image ? (
                <img
                  src={user.image}
                  alt={user.name || "User avatar"}
                  className="w-24 h-24 rounded-full border-4 border-green-500 object-cover"
                />
              ) : (
                <div className="w-24 h-24 rounded-full border-4 border-green-500 bg-green-100 dark:bg-green-800 flex items-center justify-center">
                  <span className="text-green-700 dark:text-green-300 font-bold text-2xl">
                    {user.name?.charAt(0) || user.email?.charAt(0) || "U"}
                  </span>
                </div>
              )}
            </div>

            <div className="flex-grow">
              <h2 className="text-2xl font-bold text-center md:text-left text-gray-900 dark:text-white">
                {user.name}
              </h2>
              <p className="text-gray-500 dark:text-gray-400 text-center md:text-left mb-4">
                {user.email}
              </p>

              <div className="flex flex-wrap gap-2 justify-center md:justify-start">
                <span className="px-3 py-1 bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 text-sm rounded-full">
                  Google Account
                </span>
                <span
                  className="px-3 py-1 bg-green-100 text-blue-800 dark:bg-blue-600 dark:text-blue-200 text-sm rounded-full flex items-center cursor-pointer hover:bg-green-200 dark:hover:bg-blue-700 transition-colors duration-200 gap-1.5"
                  onClick={() => {
                    handleCopyClick();
                    // alertSuccess("Mobile ID copied to clipboard!");
                  }}
                >
                  UserID: {user.mobileId || "No Mobile ID"}
                  <FiClipboard />
                </span>
                {user.role && (
                  <span
                    className={`px-3 py-1 text-sm rounded-full ${
                      user.role === "ADMIN"
                        ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                        : "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                    }`}
                  >
                    {user.role} Role
                  </span>
                )}
                <span className="px-3 py-1  text-sm rounded-full">
                  <LanguageSwitcher />
                </span>
              </div>
            </div>
          </div>
          {alertData && (
            <AleartBox
              message={alertData.message}
              type={alertData.type}
              isVisible={alertData.isVisible}
              onClose={closeAlert}
              duration={alertData.duration}
              showCloseButton={alertData.showCloseButton}
            />
          )}
        </div>

        {/* Session Manager Component */}

        <SessionManager />
      </div>
    </div>
  );
}
