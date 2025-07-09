"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useParams } from "next/navigation";
import { Link } from "@/i18n/routing";
import React, { useEffect, useState, useCallback, use } from "react";
import {
  FiArrowLeft,
  FiDownload,
  FiImage,
  FiFileText,
  FiVolume2,
  FiLoader,
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
      !batch.execFailed &&
      (!batch.isCompletedModel ||
        !batch.isCompletedDesc ||
        !batch.isCompletedAudio)
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
    if (batch.execFailed) {
      return {
        label: "Failed",
        color: "bg-red-600",
        textColor: "text-red-400",
      };
    } else if (
      batch.isCompletedModel &&
      batch.isCompletedDesc &&
      batch.isCompletedAudio
    ) {
      return {
        label: "Completed",
        color: "bg-green-600",
        textColor: "text-green-400",
      };
    } else if (batch.isCompletedModel) {
      return {
        label: "Processing",
        color: "bg-yellow-600",
        textColor: "text-yellow-400",
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
                      {languageDisplay[batch.prefferedLanguage] ||
                        batch.prefferedLanguage}
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
                        batch.isCompletedModel ? "bg-green-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.isCompletedModel
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
                        batch.isCompletedDesc ? "bg-green-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.isCompletedDesc
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
                        batch.isCompletedAudio ? "bg-green-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.isCompletedAudio
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
                        batch.execFailed ? "bg-red-500" : "bg-gray-600"
                      } mr-2`}
                    ></div>
                    <span
                      className={`${
                        batch.execFailed ? "text-red-400" : "text-gray-400"
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

              {/* Description */}
              {batch.description && (
                <div className="bg-gray-800 p-6 rounded-lg shadow-md">
                  <div className="flex items-center mb-4">
                    <FiFileText className="text-green-400 text-xl mr-2" />
                    <h2 className="text-xl font-semibold text-white">
                      Analysis Description
                    </h2>
                  </div>
                  <div className="bg-gray-900 p-4 rounded-md text-gray-300 max-h-60 overflow-y-auto">
                    {batch.description}
                  </div>
                  {batch.langDescription && (
                    <div className="mt-4">
                      <h3 className="text-lg font-semibold text-white mb-2">
                        {languageDisplay[batch.prefferedLanguage] ||
                          batch.prefferedLanguage}{" "}
                        Translation
                      </h3>
                      <div className="bg-gray-900 p-4 rounded-md text-gray-300 max-h-60 overflow-y-auto">
                        {batch.langDescription}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Audio */}
              {batch.audioURL && (
                <div className="bg-gray-800 p-6 rounded-lg shadow-md">
                  <div className="flex items-center mb-4">
                    <FiVolume2 className="text-green-400 text-xl mr-2" />
                    <h2 className="text-xl font-semibold text-white">
                      Audio Analysis
                    </h2>
                  </div>
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
              )}

              {/* If no results yet */}
              {!batch.description && !batch.audioURL && !batch.execFailed && (
                <div className="bg-gray-800 p-6 rounded-lg shadow-md text-center">
                  <FiLoader className="text-green-400 text-5xl mx-auto mb-4 animate-spin" />
                  <h2 className="text-xl font-semibold text-white mb-2">
                    Processing Your Batch
                  </h2>
                  <p className="text-gray-300">
                    Your images are being analyzed. This process may take some
                    time depending on the number of images. Check back later for
                    results.
                  </p>
                </div>
              )}

              {/* If execution failed */}
              {batch.execFailed && (
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
