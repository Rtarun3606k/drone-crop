"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useParams } from "next/navigation";
import { Link } from "@/i18n/routing";
import React, { useEffect, useState, useCallback, useMemo } from "react";
import {
  FiEye,
  FiArrowLeft,
  FiSearch,
  FiFilter,
  FiRefreshCw,
} from "react-icons/fi";
import { useTranslations } from "next-intl";
import dynamic from "next/dynamic";

export default function BatchesPage() {
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [filterCrop, setFilterCrop] = useState("");
  const [sortOrder, setSortOrder] = useState("desc"); // desc = newest first
  const router = useRouter();
  const params = useParams();
  // useParams() returns a regular object, not a Promise, so we access it directly
  const locale = params.locale;
  const session = useSession();

  // Language mapping for status display
  const languageDisplay = {
    En: "English",
    Ta: "Tamil",
    Hi: "Hindi",
    Te: "Telugu",
    Ml: "Malayalam",
    Kn: "Kannada",
  };

  // Authentication check
  useEffect(() => {
    if (session.status === "unauthenticated") {
      router.push(`/${locale}/login`);
    }
  }, [session.status, router, locale]);

  // Fetch batches
  useEffect(() => {
    if (session.status === "authenticated") {
      fetchBatches();
    }
  }, [session.status]);

  const fetchBatches = async () => {
    setLoading(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

      const response = await fetch("/api/dashboard/batches", {
        signal: controller.signal,
        headers: {
          "Cache-Control": "no-cache",
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error("Failed to fetch batches");
      }

      const data = await response.json();
      setBatches(data.batches || []);

      // Cache batches in localStorage for faster loading on subsequent visits
      try {
        localStorage.setItem(
          "cachedBatches",
          JSON.stringify({
            timestamp: Date.now(),
            data: data.batches || [],
          })
        );
      } catch (e) {
        console.warn("Could not cache batches in localStorage", e);
      }
    } catch (error) {
      console.error("Error fetching batches:", error);

      // If fetch failed, try to use cached data if available and not too old
      try {
        const cachedData = JSON.parse(localStorage.getItem("cachedBatches"));
        if (cachedData && Date.now() - cachedData.timestamp < 3600000) {
          // Less than 1 hour old
          setBatches(cachedData.data);
          setError("Using cached data. Refresh to try loading latest data.");
        } else {
          setError("Failed to load batches. Please try again.");
        }
      } catch (e) {
        setError("Failed to load batches. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Filter and sort batches
  const filteredBatches = batches
    .filter(
      (batch) =>
        batch.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
        (filterCrop === "" || batch.cropType === filterCrop)
    )
    .sort((a, b) => {
      if (sortOrder === "desc") {
        return new Date(b.createdAt) - new Date(a.createdAt);
      } else {
        return new Date(a.createdAt) - new Date(b.createdAt);
      }
    });

  // Get unique crop types for filter
  const cropTypes = [...new Set(batches.map((batch) => batch.cropType))].sort();

  // Format date nicely
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString(locale, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // Get status label and color
  const getBatchStatus = (batch) => {
    if (batch.hasExecutionFailed) {
      return { label: "Failed", color: "bg-red-600" };
    } else if (
      batch.isModelCompleted &&
      batch.isDescCompleted &&
      batch.isAudioCompleted
    ) {
      return { label: "Completed", color: "bg-green-600" };
    } else if (batch.isModelCompleted) {
      return { label: "Processing", color: "bg-yellow-600" };
    } else {
      return { label: "Pending", color: "bg-gray-600" };
    }
  };

  if (session.status === "loading") {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black pt-20 px-4 py-12">
      <div className="max-w-7xl mx-auto bg-gray-900 rounded-xl shadow-lg p-6 border border-green-500">
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
          <div className="flex items-center mb-4 md:mb-0">
            <Link
              href="/dashboard"
              className="text-green-400 hover:text-green-300 flex items-center"
            >
              <FiArrowLeft className="mr-2" />
              Back to Dashboard
            </Link>
            <h1 className="text-3xl font-bold text-green-400 ml-6">
              Your Batches
            </h1>
          </div>

          <button
            onClick={fetchBatches}
            className="flex items-center text-green-400 hover:text-green-300"
          >
            <FiRefreshCw className="mr-1" /> Refresh
          </button>
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-500 text-red-300 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        {/* Filters and search */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="flex-1 relative">
            <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search batches..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-black border border-green-500 text-white rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-400"
            />
          </div>

          <div className="md:w-1/4">
            <div className="relative">
              <FiFilter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <select
                value={filterCrop}
                onChange={(e) => setFilterCrop(e.target.value)}
                className="w-full bg-black border border-green-500 text-white rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-400 appearance-none"
              >
                <option value="">All Crops</option>
                {cropTypes.map((crop) => (
                  <option key={crop} value={crop}>
                    {crop}
                  </option>
                ))}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-white">
                <svg
                  className="fill-current h-4 w-4"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                >
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="md:w-1/4">
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value)}
              className="w-full bg-black border border-green-500 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-400"
            >
              <option value="desc">Newest First</option>
              <option value="asc">Oldest First</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
          </div>
        ) : filteredBatches.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-700">
              <thead className="bg-gray-800">
                <tr>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
                  >
                    Batch Name
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
                  >
                    Crop Type
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
                  >
                    Created
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider hidden sm:table-cell"
                  >
                    Status
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider hidden md:table-cell"
                  >
                    Language
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider hidden lg:table-cell"
                  >
                    Images
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider"
                  >
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-gray-900 divide-y divide-gray-800">
                {filteredBatches.map((batch) => {
                  const status = getBatchStatus(batch);
                  return (
                    <tr
                      key={batch.id}
                      className="hover:bg-gray-800 transition-colors"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-white">
                          {batch.name}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-300">
                          {batch.cropType}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-300">
                          {formatDate(batch.createdAt)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap hidden sm:table-cell">
                        <span
                          className={`px-2 py-1 text-xs font-semibold rounded-full text-white ${status.color}`}
                        >
                          {status.label}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap hidden md:table-cell">
                        <div className="text-sm text-gray-300">
                          {languageDisplay[batch.prefferedLanguage] ||
                            batch.prefferedLanguage}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap hidden lg:table-cell">
                        <div className="text-sm text-gray-300">
                          {batch.imagesCount}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <Link
                          href={`/dashboard/batches/${batch.id}`}
                          className="text-green-400 hover:text-green-300 flex items-center justify-end"
                        >
                          <FiEye className="mr-1" /> View
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12 bg-gray-800/30 rounded-lg">
            <p className="text-gray-300 text-lg">
              {searchTerm || filterCrop
                ? "No batches match your search criteria."
                : "You don't have any batches yet."}
            </p>
            <Link
              href="/dashboard/upload"
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
            >
              <FiEye className="mr-1" /> Upload New Batch
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
