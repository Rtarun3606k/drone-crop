"use client";

import React, { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { City,State } from "country-state-city";
// import { State, City } from "country-state-cities";

export default function ProfilePage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [form, setForm] = useState({
    phone: "",
    address: "",
    state: "",
    city: "",
  });
  const [errors, setErrors] = useState({});
  const [cities, setCities] = useState([]);

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login");
    }
  }, [status, router]);

  // Show loading state while checking authentication
  if (status === "loading") {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Don't render anything if not authenticated (will redirect via useEffect)
  if (!session) {
    return null;
  }

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

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center py-24">
      <div className="bg-white dark:bg-gray-900 shadow-lg rounded-lg p-8 w-full max-w-md border border-gray-200 dark:border-gray-800">
        <div className="flex flex-col items-center">
          {user.image ? (
            <img
              src={user.image}
              alt={user.name || "User avatar"}
              className="w-24 h-24 rounded-full border-4 border-green-500 mb-4 object-cover"
            />
          ) : (
            <div className="w-24 h-24 rounded-full border-4 border-green-500 mb-4 bg-green-100 dark:bg-green-800 flex items-center justify-center">
              <span className="text-green-700 dark:text-green-300 font-bold text-2xl">
                {user.name?.charAt(0) || user.email?.charAt(0) || "U"}
              </span>
            </div>
          )}
          <h2 className="text-2xl font-bold mb-1 text-gray-900 dark:text-white">
            {user.name}
          </h2>
          <p className="text-gray-500 dark:text-gray-400 mb-4">{user.email}</p>

          {/* Profile Form */}
          <form className="w-full space-y-4 mt-6" onSubmit={handleSubmit}>
            <div>
              <label className="block text-sm font-medium mb-1">
                Phone Number
              </label>
              <input
                type="text"
                className="w-full border rounded px-3 py-2"
                value={form.phone}
                onChange={(e) => setForm({ ...form, phone: e.target.value })}
                maxLength={10}
                placeholder="Enter 10-digit phone number"
              />
              {errors.phone && (
                <p className="text-red-500 text-xs mt-1">{errors.phone}</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Address</label>
              <input
                type="text"
                className="w-full border rounded px-3 py-2"
                value={form.address}
                onChange={(e) => setForm({ ...form, address: e.target.value })}
                placeholder="Enter your address"
              />
              {errors.address && (
                <p className="text-red-500 text-xs mt-1">{errors.address}</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">State</label>
              <select
                className="w-full border rounded px-3 py-2"
                value={form.state}
                onChange={handleStateChange}
              >
                <option value="">Select State</option>
                {allStates.map((state) => (
                  <option key={state.isoCode} value={state.name}>
                    {state.name}
                  </option>
                ))}
              </select>
              {errors.state && (
                <p className="text-red-500 text-xs mt-1">{errors.state}</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">City</label>
              <select
                className="w-full border rounded px-3 py-2"
                value={form.city}
                onChange={(e) => setForm({ ...form, city: e.target.value })}
                disabled={!form.state}
              >
                <option value="">Select City</option>
                {cities.map((city) => (
                  <option key={city} value={city}>
                    {city}
                  </option>
                ))}
              </select>
              {errors.city && (
                <p className="text-red-500 text-xs mt-1">{errors.city}</p>
              )}
            </div>
            <button
              type="submit"
              className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors"
            >
              Save Profile
            </button>
          </form>
          {/* ...existing dashboard button... */}
          <button
            className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors mt-4"
            onClick={() => router.push("/dashboard")}
          >
            Go to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
