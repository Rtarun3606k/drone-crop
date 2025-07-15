import JSZip from "jszip";
import ExifReader from "exifreader";

class GeotagChecker {
    constructor() {
        this.imageExtensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'];
    }

    // Method to check geotags in an array of File objects (for browser use)
    async checkGeotagsInFiles(files, onProgress = null) {
        try {
            if (files.length === 0) {
                return {
                    success: false,
                    message: 'No files provided',
                    results: []
                };
            }

            // Filter image files
            const imageFiles = files.filter(file => {
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                return this.imageExtensions.includes(ext);
            });

            if (imageFiles.length === 0) {
                return {
                    success: false,
                    message: 'No image files found',
                    results: []
                };
            }

            // Calculate 10% sample size (minimum 1)
            const sampleSize = Math.max(1, Math.floor(imageFiles.length * 0.1));
            const sampleFiles = this.getRandomSample(imageFiles, sampleSize);
            
            // Check geotags in sample
            const results = await this.checkGeotagsInSample(sampleFiles, onProgress);
            
            // Calculate summary
            const geotaggedCount = results.filter(r => r.hasGeotag).length;
            const errorCount = results.filter(r => r.error).length;
            const successfulChecks = results.length - errorCount;
            
            const summary = {
                totalImages: imageFiles.length,
                sampleSize: sampleFiles.length,
                successfulChecks,
                geotaggedCount,
                errorCount,
                geotagPercentage: successfulChecks > 0 ? ((geotaggedCount / successfulChecks) * 100).toFixed(1) : 0
            };
            
            return {
                success: true,
                message: `Analysis complete: ${summary.geotagPercentage}% geotagged`,
                summary,
                results
            };
            
        } catch (error) {
            return {
                success: false,
                message: `Error processing files: ${error.message}`,
                results: []
            };
        }
    }

    // Method to process a ZIP file (for browser use with File object)
    async processZipFile(zipFile, onProgress = null) {
        try {
            // Read the ZIP file
            const zip = await JSZip.loadAsync(zipFile);
            
            // Get all image files
            const imageFiles = this.getImageFiles(zip);
            
            if (imageFiles.length === 0) {
                return {
                    success: false,
                    message: 'No image files found in the ZIP file',
                    results: []
                };
            }

            // Calculate 10% sample size (minimum 1)
            const sampleSize = Math.max(1, Math.floor(imageFiles.length * 0.1));
            const sampleFiles = this.getRandomSample(imageFiles, sampleSize);
            
            // Check geotags in sample
            const results = await this.checkGeotagsInZipSample(sampleFiles, zip, onProgress);
            
            // Calculate summary
            const geotaggedCount = results.filter(r => r.hasGeotag).length;
            const errorCount = results.filter(r => r.error).length;
            const successfulChecks = results.length - errorCount;
            
            const summary = {
                totalImages: imageFiles.length,
                sampleSize: sampleFiles.length,
                successfulChecks,
                geotaggedCount,
                errorCount,
                geotagPercentage: successfulChecks > 0 ? ((geotaggedCount / successfulChecks) * 100).toFixed(1) : 0
            };
            
            return {
                success: true,
                message: `Analysis complete: ${summary.geotagPercentage}% geotagged`,
                summary,
                results
            };
            
        } catch (error) {
            return {
                success: false,
                message: `Error processing ZIP file: ${error.message}`,
                results: []
            };
        }
    }

    getImageFiles(zip) {
        const imageFiles = [];
        
        zip.forEach((relativePath, file) => {
            const ext = '.' + relativePath.split('.').pop().toLowerCase();
            if (this.imageExtensions.includes(ext) && !file.dir) {
                imageFiles.push(relativePath);
            }
        });
        
        return imageFiles;
    }

    getRandomSample(array, sampleSize) {
        const shuffled = [...array].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, sampleSize);
    }

    async checkGeotagsInSample(sampleFiles, onProgress = null) {
        const results = [];
        
        for (let i = 0; i < sampleFiles.length; i++) {
            const file = sampleFiles[i];
            const progress = ((i + 1) / sampleFiles.length) * 100;
            
            if (onProgress) {
                onProgress(progress, file.name);
            }
            
            try {
                const arrayBuffer = await file.arrayBuffer();
                const hasGeotag = await this.checkImageGeotag(arrayBuffer, file.name);
                
                results.push({
                    filename: file.name,
                    hasGeotag: hasGeotag,
                    error: null
                });
                
            } catch (error) {
                results.push({
                    filename: file.name,
                    hasGeotag: false,
                    error: error.message
                });
            }
        }
        
        return results;
    }

    async checkGeotagsInZipSample(sampleFiles, zip, onProgress = null) {
        const results = [];
        
        for (let i = 0; i < sampleFiles.length; i++) {
            const filename = sampleFiles[i];
            const progress = ((i + 1) / sampleFiles.length) * 100;
            
            if (onProgress) {
                onProgress(progress, filename.split('/').pop());
            }
            
            try {
                const file = zip.file(filename);
                const arrayBuffer = await file.async('arraybuffer');
                const hasGeotag = await this.checkImageGeotag(arrayBuffer, filename);
                
                results.push({
                    filename: filename,
                    hasGeotag: hasGeotag,
                    error: null
                });
                
            } catch (error) {
                results.push({
                    filename: filename,
                    hasGeotag: false,
                    error: error.message
                });
            }
        }
        
        return results;
    }

    async checkImageGeotag(arrayBuffer, filename) {
        try {
            const tags = ExifReader.load(arrayBuffer);
            
            // Check for GPS data
            const hasGPSLatitude = tags.GPSLatitude && tags.GPSLatitude.description;
            const hasGPSLongitude = tags.GPSLongitude && tags.GPSLongitude.description;
            const hasGPSLatitudeRef = tags.GPSLatitudeRef && tags.GPSLatitudeRef.description;
            const hasGPSLongitudeRef = tags.GPSLongitudeRef && tags.GPSLongitudeRef.description;
            
            const isGeotagged = hasGPSLatitude && hasGPSLongitude && hasGPSLatitudeRef && hasGPSLongitudeRef;
            
            return isGeotagged;
            
        } catch (error) {
            throw new Error(`Failed to read EXIF data: ${error.message}`);
        }
    }
}

export default GeotagChecker;
