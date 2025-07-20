"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useParams } from "next/navigation";
import { Link } from "@/i18n/routing";
import React, { useEffect, useState, useCallback, use } from "react";
import { generateAnalysisPDF } from "@/app/lib/pdf-generator";
import {
  FiArrowLeft,
  FiDownload,
  FiImage,
  FiFileText,
  FiVolume2,
  FiLoader,
  FiFilePlus,
} from "react-icons/fi";
import Image from "next/image";
import dynamic from "next/dynamic";

// Dynamically import the audio player component to reduce initial load time
const AudioPlayer = dynamic(() => import("react-audio-player"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-12 bg-gray-700 rounded-md animate-pulse"></div>
  ),
});

export default function BatchDetailPage({ params }) {
  // Properly unwrap params using use() before accessing properties
  const unwrappedParams = use(params);
  const id = unwrappedParams.id;

  const [batch, setBatch] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const router = useRouter();
  const routeParams = useParams();
  // useParams() returns a regular object, not a Promise, so we access it directly
  const locale = routeParams.locale;
  const session = useSession();

  // Language mapping for display
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

  // Fetch batch details
  useEffect(() => {
    if (session.status === "authenticated") {
      fetchBatchDetails();

      // Set up periodic refresh for in-progress batches
      let refreshInterval;

      return () => {
        if (refreshInterval) {
          clearInterval(refreshInterval);
        }
      };
    }
  }, [session.status, id]);

  // Set up auto-refresh for batches that are still processing
  useEffect(() => {
    let refreshInterval;

    // If batch is still processing, refresh every 30 seconds
    if (
      batch &&
      !batch.hasExecutionFailed &&
      (!batch.isModelCompleted ||
        !batch.isDescCompleted ||
        !batch.isAudioCompleted)
    ) {
      refreshInterval = setInterval(() => {
        fetchBatchDetails(true); // Silent refresh
      }, 30000); // 30 seconds
    }

    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [batch]);

  const fetchBatchDetails = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

      const response = await fetch(`/api/dashboard/batches/${id}`, {
        signal: controller.signal,
        headers: {
          "Cache-Control": "no-cache",
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error("Failed to fetch batch details");
      }

      const data = await response.json();
      console.log("Batch data received:", data.batch);
      console.log("Descriptions:", data.batch.descriptions);
      console.log("Description field:", data.batch.description);
      setBatch(data.batch);

      // Reset any error if successful
      if (error) setError("");
    } catch (error) {
      console.error("Error fetching batch details:", error);
      if (!silent) {
        setError("Failed to load batch details. Please try again.");
      }
    } finally {
      if (!silent) setLoading(false);
    }
  };

  // Generate PDF with descriptions in all languages using our utility function
  const generatePDFParent = async () => {
    try {
      // Show loading toast
      const loadingToast = document.createElement("div");
      loadingToast.className =
        "fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded shadow-lg z-50";
      loadingToast.textContent = "Generating PDF report...";
      document.body.appendChild(loadingToast);

      // Pass the batch object to the PDF generation function
      const success = await generateAnalysisPDF(
        [batch],
        languageDisplay,
        locale
      );

      document.body.removeChild(loadingToast);

      if (success) {
        // Show success toast
        const successToast = document.createElement("div");
        successToast.className =
          "fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded shadow-lg z-50";
        successToast.textContent =
          "PDF report generated! (English content fully supported)";
        document.body.appendChild(successToast);

        // Remove success toast after 3 seconds
        setTimeout(() => {
          document.body.removeChild(successToast);
        }, 3000);
      } else {
        // Show error toast
        const errorToast = document.createElement("div");
        errorToast.className =
          "fixed top-4 right-4 bg-red-600 text-white px-4 py-2 rounded shadow-lg z-50";
        errorToast.textContent = "Failed to generate PDF. Please try again.";
        document.body.appendChild(errorToast);

        // Remove error toast after 5 seconds
        setTimeout(() => {
          document.body.removeChild(errorToast);
        }, 5000);
      }
    } catch (error) {
      console.error("Error generating PDF:", error);

      const errorToast = document.createElement("div");
      errorToast.className =
        "fixed top-4 right-4 bg-red-600 text-white px-4 py-2 rounded shadow-lg z-50";
      errorToast.textContent = "Failed to generate PDF. Please try again.";
      document.body.appendChild(errorToast);

      // Remove error toast after 5 seconds
      setTimeout(() => {
        document.body.removeChild(errorToast);
      }, 5000);
    }
  };
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

  // Helper function to get a unique identifier for descriptions
  const getDescriptionKey = (desc) => {
    if (!desc) return Math.random().toString();
    // Try different ID formats
    return (
      desc.id ||
      (desc._id && typeof desc._id === "string"
        ? desc._id
        : desc._id && desc._id.$oid
        ? desc._id.$oid
        : Math.random().toString())
    );
  };

  // Get status label and color
  const getBatchStatus = (batch) => {
    if (batch.hasExecutionFailed) {
      return {
        label: "Failed",
        color: "bg-red-600",
        textColor: "text-red-400",
      };
    } else if (
      batch.isModelCompleted &&
      batch.isDescCompleted &&
      batch.isAudioCompleted
    ) {
      return {
        label: "Completed",
        color: "bg-green-600",
        textColor: "text-green-400",
      };
    } else if (
      batch.isModelCompleted ||
      (batch.descriptions && batch.descriptions.length > 0)
    ) {
      return {
        label: batch.isDescCompleted ? "Completed" : "Processing",
        color: batch.isDescCompleted ? "bg-green-600" : "bg-yellow-600",
        textColor: batch.isDescCompleted ? "text-green-400" : "text-yellow-400",
      };
    } else {
      return {
        label: "Pending",
        color: "bg-gray-600",
        textColor: "text-gray-400",
      };
    }
  };
  if (session.status === "loading" || loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black pt-20 px-4 py-12">
      <div className="max-w-7xl mx-auto bg-gray-900 rounded-xl shadow-lg p-6 border border-green-500">
        <div className="flex items-center mb-6">
          <Link
            href="/dashboard/batches"
            className="text-green-400 hover:text-green-300 flex items-center"
          >
            <FiArrowLeft className="mr-2" />
            Back to Batches
          </Link>
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-500 text-red-300 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        {batch ? (
          <div>
            <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-6 gap-4">
              <div>
                <h1 className="text-3xl font-bold text-green-400">
                  {batch.name}
                </h1>
                <p className="text-gray-300 mt-1">
                  Created on {formatDate(batch.createdAt)}
                </p>
              </div>

              {/* Status Badge */}
              {(() => {
                const status = getBatchStatus(batch);
                return (
                  <div className={`px-4 py-2 rounded-full ${status.color}`}>
                    <span className="text-white font-semibold">
                      {status.label}
                    </span>
                  </div>
                );
              })()}
            </div>

            {/* Debug Information */}
            {/* <div className="mb-6 p-4 bg-gray-800 border border-yellow-500 rounded-md text-xs overflow-auto">
              <h3 className="text-yellow-400 font-bold mb-2">
                Debug Information
              </h3>
              <div className="space-y-2 text-gray-300">
                <div>
                  <span className="text-yellow-400">Batch ID:</span> {batch.id}
                </div>
                <div>
                  <span className="text-yellow-400">
                    Has Descriptions Array:
                  </span>{" "}
                  {batch.descriptions ? "Yes" : "No"}
                </div>
                <div>
                  <span className="text-yellow-400">Descriptions Count:</span>{" "}
                  {batch.descriptions ? batch.descriptions.length : "0"}
                </div>
                <div>
                  <span className="text-yellow-400">
                    Has Legacy Description:
                  </span>{" "}
                  {batch.description ? "Yes" : "No"}
                </div>
                <div>
                  <span className="text-yellow-400">
                    Has Audio Files Array:
                  </span>{" "}
                  {batch.audioFiles ? "Yes" : "No"}
                </div>
                <div>
                  <span className="text-yellow-400">Audio Files Count:</span>{" "}
                  {batch.audioFiles ? batch.audioFiles.length : "0"}
                </div>
                <div>
                  <span className="text-yellow-400">Has Legacy Audio URL:</span>{" "}
                  {batch.audioURL ? "Yes" : "No"}
                </div>
                <div>
                  <span className="text-yellow-400">Status Flags:</span> Model:{" "}
                  {batch.isModelCompleted ? "Completed" : "Pending"}, Desc:{" "}
                  {batch.isDescCompleted ? "Completed" : "Pending"}, Audio:{" "}
                  {batch.isAudioCompleted ? "Completed" : "Pending"}, Failed:{" "}
                  {batch.hasExecutionFailed ? "Yes" : "No"}
                </div>
              </div>
            </div> */}

            {/* Batch Details */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="bg-gray-800 p-6 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold text-white mb-4 border-b border-gray-700 pb-2">
                  Batch Information
                </h2>
                <div className="space-y-3">
                  <div>
                    <span className="text-gray-400">Crop Type:</span>
                    <span className="text-white ml-2 font-medium">
                      {batch.cropType}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Images Count:</span>
                    <span className="text-white ml-2 font-medium">
                      {batch.imagesCount}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Preferred Language:</span>
                    <span className="text-white ml-2 font-medium">
                      {languageDisplay[batch.preferredLanguage] ||
                        batch.preferredLanguage}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Session ID:</span>
                    <span className="text-white ml-2 font-medium">
                      {batch.sessionId}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-800 p-6 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold text-white mb-4 border-b border-gray-700 pb-2">
                  Processing Status
                </h2>
                <div className="space-y-4">
                  <div className="flex items-center">
                    <div
                      className={`w-4 h-4 rounded-full ${
                        batch.isModelCompleted ? "bg-green-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.isModelCompleted
                          ? "text-green-400"
                          : "text-gray-400"
                      }`}
                    >
                      Model Analysis
                    </span>
                  </div>
                  <div className="flex items-center">
                    <div
                      className={`w-4 h-4 rounded-full ${
                        batch.isDescCompleted ? "bg-green-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.isDescCompleted
                          ? "text-green-400"
                          : "text-gray-400"
                      }`}
                    >
                      Description Generation
                    </span>
                  </div>
                  <div className="flex items-center">
                    <div
                      className={`w-4 h-4 rounded-full ${
                        batch.isAudioCompleted ? "bg-green-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.isAudioCompleted
                          ? "text-green-400"
                          : "text-gray-400"
                      }`}
                    >
                      Audio Generation
                    </span>
                  </div>
                  <div className="flex items-center">
                    <div
                      className={`w-4 h-4 rounded-full ${
                        batch.hasExecutionFailed ? "bg-red-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.hasExecutionFailed
                          ? "text-red-400"
                          : "text-gray-400"
                      }`}
                    >
                      Execution Failed
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Resources Section */}
            <div className="space-y-6">
              {/* Images ZIP */}
              <div className="bg-gray-800 p-6 rounded-lg shadow-md">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <FiImage className="text-green-400 text-xl mr-2" />
                    <h2 className="text-xl font-semibold text-white">
                      Image Archive
                    </h2>
                  </div>
                  <a
                    href={batch.imagesZipURL}
                    download
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 rounded-md text-white transition-colors"
                  >
                    <FiDownload className="mr-2" /> Download Images
                  </a>
                </div>
                <p className="text-gray-300 mt-2">
                  Archive containing {batch.imagesCount} drone images of{" "}
                  {batch.cropType.toLowerCase()}.
                </p>

                {/* Preview thumbnail of first image - Optimized with lazy loading */}
                {batch.imagesZipURL && batch.imagesZipURL.includes(".zip") && (
                  <div className="mt-4 flex justify-center">
                    <div className="relative w-full max-w-xs h-40 bg-gray-900 rounded-md overflow-hidden border border-gray-700">
                      <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">
                        Preview loading...
                      </div>
                      {batch.thumbnailUrl ? (
                        <Image
                          src={batch.thumbnailUrl}
                          alt="Batch thumbnail preview"
                          layout="fill"
                          objectFit="contain"
                          loading="lazy"
                        />
                      ) : (
                        <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                          <FiImage className="mr-2" /> No preview available
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Debug Info - Remove this later */}
              {/* <div className="bg-yellow-900/20 border border-yellow-600 p-4 rounded-lg mb-6">
                <h3 className="text-yellow-400 font-semibold mb-2">
                  Debug Info:
                </h3>
                <pre className="text-xs text-yellow-300 overflow-auto">
                  {JSON.stringify(
                    {
                      hasDescriptions:
                        batch.descriptions && batch.descriptions.length > 0,
                      descriptionsLength: batch.descriptions
                        ? batch.descriptions.length
                        : 0,
                      hasDescription: !!batch.description,
                      descriptionPreview: batch.description
                        ? batch.description.substring(0, 100) + "..."
                        : null,
                      descriptions: batch.descriptions,
                    },
                    null,
                    2
                  )}
                </pre>
              </div> */}

              {/* Description */}
              <div className="bg-gray-800 p-6 rounded-lg shadow-md">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <FiFileText className="text-green-400 text-xl mr-2" />
                    <h2 className="text-xl font-semibold text-white">
                      Analysis Description
                    </h2>
                  </div>
                  {batch &&
                    batch.descriptions &&
                    batch.descriptions.length > 0 && (
                      <button
                        onClick={generatePDFParent}
                        className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md text-white transition-colors"
                      >
                        <FiFilePlus className="mr-2" /> Generate PDF Report
                      </button>
                    )}
                </div>

                {/* Debug Info */}
                {/* <div className="mb-4 p-3 bg-gray-900 rounded-md border border-yellow-500">
                  <h3 className="text-yellow-500 text-sm font-semibold mb-2">
                    Debug Info:
                  </h3>
                  <pre className="text-xs text-gray-300 overflow-auto max-h-40">
                    {JSON.stringify(
                      {
                        hasDescriptions: batch?.descriptions
                          ? batch.descriptions.length > 0
                          : "none",
                        descriptionsCount: batch?.descriptions?.length || 0,
                        descriptions: batch?.descriptions || "none",
                        legacyDescription: batch?.description
                          ? "exists"
                          : "none",
                        isDescCompleted: batch?.isDescCompleted || false,
                        batchId: batch?.id,
                      },
                      null,
                      2
                    )}
                  </pre>
                </div> */}

                {/* Debug information */}
                {/* <div className="bg-gray-900 p-4 mb-4 rounded border border-gray-700 text-xs font-mono">
                  <div className="text-yellow-400">Debug Info:</div>
                  <div className="text-gray-400">
                    Has batch: {batch ? "Yes" : "No"}
                  </div>
                  <div className="text-gray-400">
                    Has descriptions:{" "}
                    {batch && batch.descriptions
                      ? `Yes (${batch.descriptions.length})`
                      : "No"}
                  </div>
                  <div className="text-gray-400">
                    Batch ID: {batch ? batch.id : "N/A"}
                  </div>
                </div> */}

                {/* Display descriptions by language */}
                {batch &&
                batch.descriptions &&
                batch.descriptions.length > 0 ? (
                  batch.descriptions.map((desc) => (
                    <div
                      key={getDescriptionKey(desc)}
                      className="mb-6 last:mb-0 border-b border-gray-700 last:border-b-0 pb-6 last:pb-0"
                    >
                      <h3 className="text-lg font-semibold text-white mb-4">
                        {languageDisplay[desc.language] || desc.language}
                      </h3>

                      {/* Briefed (Long Description) */}
                      <div className="mb-4">
                        <h4 className="text-md font-medium text-green-400 mb-2">
                          📝 Briefed Analysis
                        </h4>
                        <div className="bg-gray-900 p-4 rounded-md text-gray-300 max-h-60 overflow-y-auto border-l-4 border-green-500">
                          {desc.longDescription}
                        </div>
                      </div>

                      {/* Summarised (Short Description) */}
                      {desc.shortDescription && (
                        <div className="mb-4">
                          <h4 className="text-md font-medium text-blue-400 mb-2">
                            📋 Summarised Analysis
                          </h4>
                          <div className="bg-gray-900 p-4 rounded-md text-gray-300 border-l-4 border-blue-500">
                            {desc.shortDescription}
                          </div>
                        </div>
                      )}

                      {/* Metadata */}
                      {(desc.wordCount || desc.confidence) && (
                        <div className="text-xs text-gray-400 mt-2 flex gap-4">
                          {desc.wordCount && (
                            <span>📊 Words: {desc.wordCount}</span>
                          )}
                          {desc.confidence && (
                            <span>
                              🎯 Confidence:{" "}
                              {(desc.confidence * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  ))
                ) : batch && batch.description ? (
                  /* Legacy description support */
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-4">
                      Description
                    </h3>
                    <div className="mb-4">
                      <h4 className="text-md font-medium text-green-400 mb-2">
                        📝 Briefed Analysis
                      </h4>
                      <div className="bg-gray-900 p-4 rounded-md text-gray-300 max-h-60 overflow-y-auto border-l-4 border-green-500">
                        {batch.description}
                      </div>
                    </div>
                  </div>
                ) : (
                  /* No descriptions available */
                  <div className="text-center py-8">
                    <FiFileText className="text-gray-600 text-4xl mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-400 mb-2">
                      No Analysis Available
                    </h3>
                    <p className="text-gray-500">
                      Analysis description will appear here once processing is
                      complete.
                    </p>
                  </div>
                )}
              </div>

              {/* Audio */}
              <div className="bg-gray-800 p-6 rounded-lg shadow-md">
                <div className="flex items-center mb-4">
                  <FiVolume2 className="text-green-400 text-xl mr-2" />
                  <h2 className="text-xl font-semibold text-white">
                    Audio Analysis
                  </h2>
                </div>

                {/* Display audio files by language */}
                {batch && batch.audioFiles && batch.audioFiles.length > 0 ? (
                  batch.audioFiles.map((audio) => (
                    <div key={audio.id} className="mb-6 last:mb-0">
                      <h3 className="text-lg font-semibold text-white mb-2">
                        {languageDisplay[audio.language] || audio.language}
                      </h3>
                      <div className="bg-gray-900 p-4 rounded-md">
                        <audio controls className="w-full">
                          <source src={audio.fileUrl} type="audio/mpeg" />
                          Your browser does not support the audio element.
                        </audio>
                        <div className="mt-3 flex justify-between items-center">
                          <div className="text-xs text-gray-400">
                            {audio.duration && (
                              <span>Duration: {audio.duration}s</span>
                            )}
                            {audio.fileSize && (
                              <span className="ml-4">
                                Size:{" "}
                                {(audio.fileSize / 1024 / 1024).toFixed(2)} MB
                              </span>
                            )}
                          </div>
                          <a
                            href={audio.fileUrl}
                            download
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center px-3 py-1 bg-green-600 hover:bg-green-700 rounded-md text-sm text-white transition-colors"
                          >
                            <FiDownload className="mr-1" /> Download Audio
                          </a>
                        </div>
                      </div>
                    </div>
                  ))
                ) : batch && batch.audioURL ? (
                  /* Legacy audio support */
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-2">
                      Audio
                    </h3>
                    <div className="bg-gray-900 p-4 rounded-md">
                      <audio controls className="w-full">
                        <source src={batch.audioURL} type="audio/mpeg" />
                        Your browser does not support the audio element.
                      </audio>
                      <div className="mt-3 flex justify-end">
                        <a
                          href={batch.audioURL}
                          download
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center px-3 py-1 bg-green-600 hover:bg-green-700 rounded-md text-sm text-white transition-colors"
                        >
                          <FiDownload className="mr-1" /> Download Audio
                        </a>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* No audio available */
                  <div className="text-center py-8">
                    <FiVolume2 className="text-gray-600 text-4xl mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-400 mb-2">
                      No Audio Available
                    </h3>
                    <p className="text-gray-500">
                      Audio analysis will appear here once processing is
                      complete.
                    </p>
                  </div>
                )}
              </div>

              {/* If no results yet */}
              {!batch.description &&
                (!batch.descriptions || batch.descriptions.length === 0) &&
                !batch.audioURL &&
                (!batch.audioFiles || batch.audioFiles.length === 0) &&
                !batch.hasExecutionFailed && (
                  <div className="bg-gray-800 p-6 rounded-lg shadow-md text-center">
                    <FiLoader className="text-green-400 text-5xl mx-auto mb-4 animate-spin" />
                    <h2 className="text-xl font-semibold text-white mb-2">
                      Processing Your Batch
                    </h2>
                    <p className="text-gray-300">
                      Your images are being analyzed. This process may take some
                      time depending on the number of images. Check back later
                      for results.
                    </p>
                  </div>
                )}

              {/* If execution failed */}
              {batch.hasExecutionFailed && (
                <div className="bg-red-900/20 border border-red-600 p-6 rounded-lg shadow-md">
                  <h2 className="text-xl font-semibold text-red-400 mb-2">
                    Processing Failed
                  </h2>
                  <p className="text-gray-300">
                    There was an error processing your batch. Please contact
                    support for assistance.
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-300 text-lg">
              Batch not found or you don't have permission to view it.
            </p>
            <Link
              href="/dashboard/batches"
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
            >
              <FiArrowLeft className="mr-1" /> Back to All Batches
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
