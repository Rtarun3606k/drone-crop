"use client";

import { auth } from "@/app/auth";
import { Link, redirect, useRouter } from "@/i18n/routing";
import React, { useState } from "react";
import { useTranslations } from "next-intl";
import { FiUpload, FiX } from "react-icons/fi";
import Image from "next/image";
import JSZip from "jszip";
import { useSession } from "next-auth/react";
import { Param } from "@/app/generated/prisma/runtime/library";
import { useParams } from "next/navigation";

// List of crops for dropdown
const cropOptions = [
  "Soybean",
  "Rice",
  "Wheat",
  "Maize",
  "Cotton",
  "Sugarcane",
  "Potato",
  "Tomato",
  "Chili",
  "Other",
];

export default function UploadPage() {
  const [batchName, setBatchName] = useState("");
  const [selectedCrop, setSelectedCrop] = useState("Soybean");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [previewImages, setPreviewImages] = useState([]);
  const [formError, setFormError] = useState("");
  const router = useRouter();
  const params = useParams();
  // useParams() returns a regular object, not a Promise, so we access it directly
  const locale = params.locale;
  // Initialize state with the locale value directly
  const [defaultsetLang, setdefaultSetLang] = useState(locale);

  // Get translations for upload namespace
  const t = useTranslations("upload");

  const session = useSession();

  // Use useEffect to handle redirection after component mount
  React.useEffect(() => {
    if (session.status === "unauthenticated") {
      router.push(`/${locale}/login`);
    }
  }, [session.status, router, locale]);

  // Debug function to see the current locale value
  const getLocale = React.useCallback(() => {
    console.log("Current locale:", locale);
    console.log("Default set lang:", defaultsetLang);
  }, [locale, defaultsetLang]);

  // Using locale from params instead of manually parsing the URL
  // Handle file selection
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);

    // Validate file types (only images)
    const validFiles = files.filter((file) => file.type.startsWith("image/"));

    if (validFiles.length !== files.length) {
      setFormError(t("form_error_images_only"));
      return;
    }

    setSelectedFiles((prev) => [...prev, ...validFiles]);

    // Create preview URLs for the selected images
    const newPreviewImages = validFiles.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
    }));

    setPreviewImages((prev) => [...prev, ...newPreviewImages]);
    setFormError("");
  };

  // Remove a selected file
  const removeFile = (index) => {
    const updatedFiles = [...selectedFiles];
    updatedFiles.splice(index, 1);
    setSelectedFiles(updatedFiles);

    // Also remove the preview
    const updatedPreviews = [...previewImages];
    URL.revokeObjectURL(updatedPreviews[index].preview);
    updatedPreviews.splice(index, 1);
    setPreviewImages(updatedPreviews);
  }; // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Form validation
    if (!batchName.trim()) {
      setFormError(t("form_error_batch_name"));
      return;
    }

    if (selectedFiles.length === 0) {
      setFormError(t("form_error_select_image"));
      return;
    }

    // Warning for large batches
    if (selectedFiles.length > 100) {
      const confirmed = window.confirm(
        t("form_warning_large_batch", { count: selectedFiles.length })
      );
      if (!confirmed) return;
    }

    // Simulate form submission
    setIsUploading(true);

    try {
      // Here you would typically send the data to your backend
      // For now, we'll just simulate a delay
      //   await new Promise((resolve) => setTimeout(resolve, 2000));
      const zip = new JSZip();
      selectedFiles.forEach((file, index) => {
        console.log("Adding file to zip:", file.name || `image-${index}.jpg`);
        zip.file(file.name || `image-${index}.jpg`, file);
      });

      const zipBlob = await zip.generateAsync({ type: "blob" });

      const formData = new FormData();
      formData.append("batchName", batchName);
      formData.append("cropType", selectedCrop);
      formData.append(
        "imagesZip",
        zipBlob,
        `${batchName + session.data.user.email + Date.now()}.zip`
      );
      formData.append("defaultSetLang", defaultsetLang);
      formData.append("imagesCount", selectedFiles.length);
      console.log("Form data prepared for upload:", {
        batchName,
        cropType: selectedCrop,
        imagesCount: selectedFiles.length,
      });

      const request = await fetch("/api/dashboard/upload", {
        method: "POST",
        body: formData,
      });

      if (!request.ok) {
        throw new Error("Network response was not ok");
      }
      const response = await request.json();
      console.log("Upload response:", response);

      // Reset form after successful submission
      //   setBatchName("");
      //   setSelectedCrop("Soybean");
      //   setSelectedFiles([]);
      //   setPreviewImages([]);
      //   setFormError("");

      // Show success message or redirect
      alert(t("success"));
    } catch (error) {
      console.error("Upload failed:", error);
      setFormError(t("form_error_upload_failed"));
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black pt-20 px-4 py-12">
      <div className="max-w-3xl mx-auto bg-gray-900 rounded-xl shadow-lg p-6 border border-green-500">
        <h1 className="text-3xl font-bold text-green-400 mb-6 text-center">
          {t("title")}
        </h1>

        {formError && (
          <div className="bg-red-900/30 border border-red-500 text-red-300 px-4 py-3 rounded mb-6">
            {formError}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Batch Name Field */}
          <div>
            <label
              htmlFor="batchName"
              className="block text-white font-semibold mb-2"
            >
              {t("batch_name_label")}
            </label>
            <input
              type="text"
              id="batchName"
              value={batchName}
              onChange={(e) => setBatchName(e.target.value)}
              placeholder={t("batch_name_placeholder")}
              className="w-full bg-black border border-green-500 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-400"
              required
            />
          </div>

          {/* Crop Selection Field */}
          <div>
            <label
              htmlFor="cropType"
              className="block text-white font-semibold mb-2"
            >
              {t("crop_type_label")}
            </label>
            <select
              id="cropType"
              value={selectedCrop}
              onChange={(e) => setSelectedCrop(e.target.value)}
              className="w-full bg-black border border-green-500 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-400"
            >
              {cropOptions.map((crop) => (
                <option key={crop} value={crop}>
                  {crop}
                </option>
              ))}
            </select>
          </div>

          {/* Image Upload Field */}
          <div>
            <label className="block text-white font-semibold mb-2">
              {t("upload_label")}
            </label>
            <div
              className="border-2 border-dashed border-green-500 rounded-lg p-8 text-center cursor-pointer hover:bg-green-900/20 transition-colors"
              onClick={() => document.getElementById("fileInput").click()}
            >
              <FiUpload className="mx-auto h-10 w-10 text-green-400" />
              <p className="text-white mt-2">
                {t("upload_hint")}
              </p>
              <p className="text-gray-400 text-sm mt-1">
                {t("upload_formats")}
              </p>
              {selectedFiles.length > 0 && (
                <p className="text-green-400 font-medium mt-2">
                  {t("selected_files", { count: selectedFiles.length })}
                </p>
              )}
              <input
                type="file"
                id="fileInput"
                multiple
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>
          </div>

          {/* Preview Selected Images */}
          {previewImages.length > 0 && (
            <div>
              <h3 className="text-white font-semibold mb-3">
                {t("selected_images", { count: previewImages.length })}
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                {/* Only show the first 5 images for preview */}
                {previewImages.slice(0, 5).map((image, index) => (
                  <div key={index} className="relative group">
                    <div className="aspect-square rounded-lg overflow-hidden border border-gray-700">
                      <img
                        src={image.preview}
                        alt={`Preview ${index}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFile(index)}
                      className="absolute -top-2 -right-2 bg-red-600 text-white rounded-full p-1 hover:bg-red-700"
                    >
                      <FiX size={16} />
                    </button>
                  </div>
                ))}

                {/* Show a message if there are more images than shown */}
                {previewImages.length > 5 && (
                  <div className="aspect-square rounded-lg flex items-center justify-center border border-gray-700 bg-gray-800/50">
                    <p className="text-white text-center font-medium">
                      {t("more_images", { count: previewImages.length - 5 })}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Submit Button */}
          <div className="flex justify-center pt-4">
            <button
              type="submit"
              disabled={isUploading}
              className={`px-6 py-3 rounded-lg bg-green-600 text-white font-semibold hover:bg-green-700 transition-colors ${
                isUploading ? "opacity-70 cursor-not-allowed" : ""
              }`}
            >
              {isUploading ? t("submit_uploading") : t("submit")}
            </button>
          </div>
        </form>

        {/* Back to Dashboard Link */}
        <div className="mt-8 text-center">
          <Link
            href="/dashboard"
            className="text-green-400 hover:text-green-300 underline"
          >
            {t("back_to_dashboard")}
          </Link>
        </div>
      </div>
    </div>
  );
}
