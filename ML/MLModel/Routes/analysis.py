import json
from  Routes.Model import batch_predict

def analyze_batch_characteristics(results):
    """
    Analyze batch prediction results and provide overall characteristics
    
    Parameters:
    results (list): List of prediction results from batch_predict
    
    Returns:
    dict: Comprehensive analysis of the batch predictions
    """
    if not results:
        return {"error": "No results to analyze"}
    
    # Initialize counters and accumulators
    total_images = len(results)
    successful_predictions = 0
    failed_predictions = 0
    class_counts = {}
    confidence_scores = []
    all_probabilities = {}
    
    # Class names for soybean disease detection
    class_names = ['Healthy_Soyabean', 'Soyabean Semilooper_Pest_Attack', 'Soyabean_Mosaic', 'rust']
    
    # Initialize probability accumulators
    for class_name in class_names:
        all_probabilities[class_name] = []
        class_counts[class_name] = 0
    
    error_details = []
    
    # Process each result
    for result in results:
        if result['status'] == 'success':
            successful_predictions += 1
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            # Count class predictions
            if predicted_class in class_counts:
                class_counts[predicted_class] += 1
            
            # Collect confidence scores
            confidence_scores.append(confidence)
            
            # Collect probabilities for each class
            for class_name in class_names:
                if class_name in result['probabilities']:
                    all_probabilities[class_name].append(result['probabilities'][class_name])
        
        else:
            failed_predictions += 1
            error_details.append({
                'image': result.get('image_name', 'Unknown'),
                'error': result.get('error', 'Unknown error')
            })
    
    # Calculate statistics
    success_rate = (successful_predictions / total_images) * 100 if total_images > 0 else 0
    
    # Confidence statistics
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        min_confidence = min(confidence_scores)
        max_confidence = max(confidence_scores)
        
        # Confidence distribution
        high_confidence = len([c for c in confidence_scores if c >= 80])
        medium_confidence = len([c for c in confidence_scores if 60 <= c < 80])
        low_confidence = len([c for c in confidence_scores if c < 60])
    else:
        avg_confidence = min_confidence = max_confidence = 0
        high_confidence = medium_confidence = low_confidence = 0
    
    # Class distribution and average probabilities
    class_percentages = {}
    avg_class_probabilities = {}
    
    for class_name in class_names:
        class_percentages[class_name] = (class_counts[class_name] / successful_predictions * 100) if successful_predictions > 0 else 0
        avg_class_probabilities[class_name] = (sum(all_probabilities[class_name]) / len(all_probabilities[class_name])) if all_probabilities[class_name] else 0
    
    # Determine batch health status
    healthy_percentage = class_percentages.get('Healthy_Soyabean', 0)
    if healthy_percentage >= 80:
        batch_health_status = "Excellent"
        health_description = "Crop is in excellent condition with minimal disease presence"
    elif healthy_percentage >= 60:
        batch_health_status = "Good"
        health_description = "Crop is in good condition with some disease presence"
    elif healthy_percentage >= 40:
        batch_health_status = "Moderate"
        health_description = "Crop shows moderate disease presence, monitoring recommended"
    elif healthy_percentage >= 20:
        batch_health_status = "Poor"
        health_description = "Crop shows significant disease presence, intervention needed"
    else:
        batch_health_status = "Critical"
        health_description = "Crop is severely affected by diseases, immediate action required"
    
    # Identify primary concerns
    disease_classes = {k: v for k, v in class_percentages.items() if k != 'Healthy_Soyabean' and v > 0}
    primary_disease = max(disease_classes.items(), key=lambda x: x[1]) if disease_classes else None
    
    # Create comprehensive analysis
    analysis = {
        "batch_summary": {
            "total_images": total_images,
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "success_rate": round(success_rate, 2)
        },
        
        "confidence_analysis": {
            "average_confidence": round(avg_confidence, 2),
            "min_confidence": round(min_confidence, 2),
            "max_confidence": round(max_confidence, 2),
            "confidence_distribution": {
                "high_confidence_80_plus": high_confidence,
                "medium_confidence_60_79": medium_confidence,
                "low_confidence_below_60": low_confidence
            }
        },
        
        "class_distribution": {
            "counts": class_counts,
            "percentages": {k: round(v, 2) for k, v in class_percentages.items()}
        },
        
        "average_probabilities": {k: round(v, 2) for k, v in avg_class_probabilities.items()},
        
        "health_assessment": {
            "overall_status": batch_health_status,
            "healthy_percentage": round(healthy_percentage, 2),
            "description": health_description
        },
        
        "disease_analysis": {
            "primary_disease": primary_disease[0] if primary_disease else None,
            "primary_disease_percentage": round(primary_disease[1], 2) if primary_disease else 0,
            "diseases_detected": [k for k, v in disease_classes.items() if v > 5]  # Diseases affecting >5% of crop
        },
        
        "recommendations": generate_recommendations(class_percentages, avg_confidence),
        
        "errors": {
            "count": failed_predictions,
            "details": error_details[:5] if len(error_details) > 5 else error_details  # Limit to first 5 errors
        }
    }
    
    return analysis

def generate_recommendations(class_percentages, avg_confidence):
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    healthy_pct = class_percentages.get('Healthy_Soyabean', 0)
    mosaic_pct = class_percentages.get('Soyabean_Mosaic', 0)
    pest_pct = class_percentages.get('Soyabean Semilooper_Pest_Attack', 0)
    rust_pct = class_percentages.get('rust', 0)
    
    # Health-based recommendations
    if healthy_pct < 50:
        recommendations.append("Immediate field inspection recommended due to high disease prevalence")
    
    if mosaic_pct > 20:
        recommendations.append("Apply virus-resistant varieties and control aphid vectors for mosaic management")
    
    if pest_pct > 15:
        recommendations.append("Implement integrated pest management for semilooper control")
    
    if rust_pct > 10:
        recommendations.append("Apply appropriate fungicides and ensure proper field drainage for rust control")
    
    # Confidence-based recommendations
    if avg_confidence < 70:
        recommendations.append("Consider manual verification due to lower prediction confidence")
    
    if not recommendations:
        recommendations.append("Continue current management practices and regular monitoring")
    
    return recommendations

# Example usage function to integrate with your batch prediction
def process_batch_with_analysis(image_paths, model_path, num_threads=4, output_json_path=None):
    """
    Process batch predictions and generate comprehensive analysis
    
    Returns both raw results and analysis
    """
    # Get batch predictions
    results = batch_predict(image_paths, model_path, num_threads, output_json_path)
    
    # Analyze the results
    analysis = analyze_batch_characteristics(results)
    
    # Save analysis if output path provided
    if output_json_path:
        analysis_path = output_json_path.replace('.json', '_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {analysis_path}")
    
    return results, analysis

# Function to generate LLM-ready summary
def generate_llm_summary(analysis):
    """
    Generate a text summary suitable for LLM processing
    
    Parameters:
    analysis (dict): Analysis results from analyze_batch_characteristics
    
    Returns:
    str: Formatted text summary for LLM input
    """
    summary = f"""
CROP HEALTH ANALYSIS REPORT

BATCH OVERVIEW:
- Total Images Analyzed: {analysis['batch_summary']['total_images']}
- Success Rate: {analysis['batch_summary']['success_rate']}%
- Average Prediction Confidence: {analysis['confidence_analysis']['average_confidence']}%

HEALTH STATUS: {analysis['health_assessment']['overall_status']}
- Healthy Crop: {analysis['health_assessment']['healthy_percentage']}%
- {analysis['health_assessment']['description']}

DISEASE DISTRIBUTION:
"""
    
    for class_name, percentage in analysis['class_distribution']['percentages'].items():
        summary += f"- {class_name}: {percentage}%\n"
    
    if analysis['disease_analysis']['primary_disease']:
        summary += f"\nPRIMARY CONCERN: {analysis['disease_analysis']['primary_disease']} ({analysis['disease_analysis']['primary_disease_percentage']}%)\n"
    
    summary += "\nRECOMMENDATIONS:\n"
    for i, rec in enumerate(analysis['recommendations'], 1):
        summary += f"{i}. {rec}\n"
    
    return summary.strip()