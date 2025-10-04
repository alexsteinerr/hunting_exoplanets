from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Mock data - in a real app, this would come from a database or API
planet_data = {
    "mercury": {
        "is_planet": True,
        "type": "Terrestrial",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "venus": {
        "is_planet": True,
        "type": "Terrestrial",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "earth": {
        "is_planet": True,
        "type": "Terrestrial",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "mars": {
        "is_planet": True,
        "type": "Terrestrial",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "jupiter": {
        "is_planet": True,
        "type": "Gas Giant",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "saturn": {
        "is_planet": True,
        "type": "Gas Giant",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "uranus": {
        "is_planet": True,
        "type": "Ice Giant",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "neptune": {
        "is_planet": True,
        "type": "Ice Giant",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    },
    "pluto": {
        "is_planet": False,
        "type": "Dwarf Planet",
        "diameter": "1",
        "mass": "1",
        "distance_from_sun": "1",
        "orbital_period": "1",
        "description": "1"
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_planet():
    planet_name = request.form.get('planet_name', '').lower().strip()
    
    if planet_name in planet_data:
        return jsonify(planet_data[planet_name])
    else:
        return jsonify({
            "is_planet": False,
            "type": "Unknown",
            "diameter": "1",
            "mass": "1",
            "distance_from_sun": "1",
            "orbital_period": "1",
            "description": "1"
        })

if __name__ == '__main__':
    app.run(debug=True)