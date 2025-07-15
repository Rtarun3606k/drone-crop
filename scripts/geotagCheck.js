const fs = require('fs');
const path = require('path');
const JSZip = require('jszip');
const ExifReader = require('exifreader');

class GeotagChecker {
    constructor() {
        this.imageExtensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'];
    }

    async processZipFile(zipFilePath) {
        try {
            console.log(`🔍 Processing ZIP file: ${zipFilePath}`);
            
            // Read the ZIP file
            const zipBuffer = fs.readFileSync(zipFilePath);
            const zip = await JSZip.loadAsync(zipBuffer);
            
            // Get all image files
            const imageFiles = this.getImageFiles(zip);
            
            if (imageFiles.length === 0) {
                console.log('❌ No image files found in the ZIP file');
                return;
            }

            // Calculate 10% sample size (minimum 1)
            const sampleSize = Math.max(1, Math.floor(imageFiles.length * 0.1));
            const sampleFiles = this.getRandomSample(imageFiles, sampleSize);
            
            console.log(`📊 Total images found: ${imageFiles.length}`);
            console.log(`🎯 Testing sample size (10%): ${sampleFiles.length}`);
            console.log('─'.repeat(60));
            
            // Check geotags in sample
            const results = await this.checkGeotagsInSample(sampleFiles, zip);
            
            // Display results
            this.displayResults(results, imageFiles.length, sampleFiles.length);
            
        } catch (error) {
            console.error('❌ Error processing ZIP file:', error.message);
        }
    }

    getImageFiles(zip) {
        const imageFiles = [];
        
        zip.forEach((relativePath, file) => {
            const ext = path.extname(relativePath).toLowerCase();
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

    async checkGeotagsInSample(sampleFiles, zip) {
        const results = [];
        
        console.log('🔄 Starting geotag analysis...\n');
        
        for (let i = 0; i < sampleFiles.length; i++) {
            const filename = sampleFiles[i];
            const progress = ((i + 1) / sampleFiles.length) * 100;
            
            console.log(`[${Math.round(progress)}%] Checking: ${path.basename(filename)}`);
            
            try {
                const file = zip.file(filename);
                const arrayBuffer = await file.async('arraybuffer');
                const hasGeotag = await this.checkImageGeotag(arrayBuffer, filename);
                
                results.push({
                    filename: filename,
                    hasGeotag: hasGeotag,
                    error: null
                });
                
                console.log(`   ${hasGeotag ? '✅ Geotagged' : '❌ Not geotagged'}`);
                
            } catch (error) {
                console.log(`   ⚠️ Error: ${error.message}`);
                results.push({
                    filename: filename,
                    hasGeotag: false,
                    error: error.message
                });
            }
            
            console.log(''); // Empty line for readability
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
            
            // Log GPS coordinates if found
            if (isGeotagged) {
                const lat = tags.GPSLatitude.description;
                const lon = tags.GPSLongitude.description;
                const latRef = tags.GPSLatitudeRef.description;
                const lonRef = tags.GPSLongitudeRef.description;
                console.log(`   📍 GPS: ${lat}${latRef}, ${lon}${lonRef}`);
            }
            
            return isGeotagged;
            
        } catch (error) {
            throw new Error(`Failed to read EXIF data: ${error.message}`);
        }
    }

    displayResults(results, totalImages, sampleSize) {
        const geotaggedCount = results.filter(r => r.hasGeotag).length;
        const errorCount = results.filter(r => r.error).length;
        const successfulChecks = results.length - errorCount;
        
        console.log('═'.repeat(60));
        console.log('📋 ANALYSIS SUMMARY');
        console.log('═'.repeat(60));
        console.log(`📁 Total images in ZIP: ${totalImages}`);
        console.log(`🎯 Sample size tested (10%): ${sampleSize}`);
        console.log(`✅ Successfully analyzed: ${successfulChecks}`);
        console.log(`📍 Geotagged images found: ${geotaggedCount}/${successfulChecks}`);
        console.log(`⚠️ Errors encountered: ${errorCount}`);
        
        if (successfulChecks > 0) {
            const percentage = ((geotaggedCount / successfulChecks) * 100).toFixed(1);
            console.log(`📊 Geotag percentage: ${percentage}%`);
        }
        
        console.log('═'.repeat(60));
        
        // Detailed results
        console.log('\n📝 DETAILED RESULTS:');
        console.log('─'.repeat(60));
        
        results.forEach((result, index) => {
            const filename = path.basename(result.filename);
            const status = result.error ? '⚠️ ERROR' : 
                          result.hasGeotag ? '✅ GEOTAGGED' : '❌ NOT GEOTAGGED';
            
            console.log(`${index + 1}. ${filename}`);
            console.log(`   Status: ${status}`);
            
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }
            
            console.log('');
        });
        
        
    }
}

// Main execution
async function main() {
    // Dummy ZIP file path - replace with actual path
    const zipFilePath = '/home/bruh/Downloads/test.zip';
    
    console.log('🚀 Image Geotag Checker Starting...');
    console.log('═'.repeat(60));
    
    // Check if ZIP file exists
    if (!fs.existsSync(zipFilePath)) {
        console.log(`❌ ZIP file not found: ${zipFilePath}`);
        console.log('📝 Please update the zipFilePath variable with the correct path to your ZIP file.');
        return;
    }
    
    const checker = new GeotagChecker();
    await checker.processZipFile(zipFilePath);
    
    console.log('🏁 Analysis complete!');
}

// Run the script
main().catch(console.error);

// Export for use as module
module.exports = GeotagChecker;