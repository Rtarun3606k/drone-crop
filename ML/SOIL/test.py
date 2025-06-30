import requests

def get_soil_data(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=sand&property=silt&property=clay&depth=0-5cm"
    response = requests.get(url)
    data = response.json()

    try:
        layers = data["properties"]["layers"]
        get_mean = lambda name: next(
            (d["depths"][0]["values"]["mean"] for d in layers if d["name"] == name),
            None
        )

        sand = get_mean("sand")
        silt = get_mean("silt")
        clay = get_mean("clay")

        if None in (sand, silt, clay):
            raise ValueError("One or more soil values are missing (None).")

        # Convert g/kg to percentage
        sand /= 10
        silt /= 10
        clay /= 10

        total = sand + silt + clay
        sand = round(sand / total * 100, 1)
        silt = round(silt / total * 100, 1)
        clay = round(clay / total * 100, 1)

        print(f"Sand: {sand}%, Silt: {silt}%, Clay: {clay}%")
        print("🧪 Soil Texture:", classify_soil(sand, silt, clay))
        print("Available layers:")


    except Exception as e:
        for layer in data["properties"]["layers"]:
            print(layer["name"], layer["depths"][0]["values"])
        print("❌ Failed:", e)

def classify_soil(sand, silt, clay):
    if clay > 40:
        return "Clay"
    elif clay > 27 and silt > 28 and silt < 40:
        return "Clay Loam"
    elif silt >= 80 and clay < 12:
        return "Silt"
    elif silt >= 50 and clay < 27:
        return "Silty Loam"
    elif sand >= 70 and clay < 15:
        return "Sandy Loam"
    elif sand >= 85 and clay < 10:
        return "Sand"
    else:
        return "Loam"

# Example usage
get_soil_data(12.9716, 77.5946)
# Example coordinates for Bangalore, India
get_soil_data(28.6139, 77.2090)  # Delhi
get_soil_data(19.0760, 72.8777)  # Mumbai
get_soil_data(22.5726, 88.3639)  # Kolkata
