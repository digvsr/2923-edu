from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from deap import base, creator, tools
import random

app = Flask(__name__)
CORS(app)

data = pd.read_csv("C:/Users/digvi/syntac/back-end/sample.csv")

# Genetic algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_timetable(class_name, section, day):
    class_data = data[(data["class_name"] == class_name) & 
                      (data["section"] == section) & 
                      (data["day_of_week"] == day) & 
                      (data["teacher_availability"] == "Yes")]
    
    time_slots = ["9:00 AM - 10:00 AM", "10:00 AM - 11:00 AM", "11:00 AM - 12:00 PM", 
                  "12:00 PM - 1:00 PM"]
    
    def create_individual():
        assignments = []
        for slot in time_slots:
            available = class_data[class_data["time_slot"] == slot]
            if not available.empty:
                row = available.iloc[random.randint(0, len(available) - 1)]
                assignments.append((row["teacher_id"], row["subject"], slot, row["room_number"]))
        return assignments

    def evaluate(individual):
        conflicts = 0
        seen_teachers = set()
        seen_rooms = set()
        for teacher_id, _, slot, room in individual:
            teacher_key = (teacher_id, slot)
            room_key = (room, slot)
            if teacher_key in seen_teachers or room_key in seen_rooms:
                conflicts += 1
            seen_teachers.add(teacher_key)
            seen_rooms.add(room_key)
        return conflicts,

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    for _ in range(30):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        tools.cxTwoPoint(offspring[::2], offspring[1::2])
        for mutant in offspring:
            if random.random() < 0.1:
                toolbox.mutate(mutant)
        pop[:] = offspring
        fits = [toolbox.evaluate(ind) for ind in pop]
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

    best = tools.selBest(pop, 1)[0]
    return pd.DataFrame(best, columns=["teacher_id", "subject", "time_slot", "room_number"])

def adjust_timetable(timetable, unavailable_teacher):
    df = pd.DataFrame(timetable)
    conflicts = df[df["teacher_id"] == unavailable_teacher]
    if conflicts.empty:
        return df
    
    class_name = request.args.get("class_name")  # Get from query params
    section = request.args.get("section")
    day = request.args.get("day", "Monday")
    
    for _, conflict in conflicts.iterrows():
        df = df[~((df["teacher_id"] == unavailable_teacher) & 
                  (df["time_slot"] == conflict["time_slot"]))]
        available = data[(data["class_name"] == class_name) & 
                        (data["section"] == section) & 
                        (data["day_of_week"] == day) & 
                        (data["teacher_availability"] == "Yes") & 
                        (~data["teacher_id"].isin(df["teacher_id"])) & 
                        (data["time_slot"] == conflict["time_slot"])]
        if not available.empty:
            row = available.iloc[0]
            df = pd.concat([df, pd.DataFrame([{"teacher_id": row["teacher_id"], 
                                              "subject": row["subject"], 
                                              "time_slot": conflict["time_slot"], 
                                              "room_number": conflict["room_number"]}])], 
                          ignore_index=True)
    return df

@app.route('/generate', methods=['POST'])
def generate():
    req = request.get_json()
    if not req or "class_name" not in req or "section" not in req:
        return jsonify({"error": "Missing class_name or section"}), 400
    class_name = req["class_name"]
    section = req["section"]
    day = req.get("day", "Monday")
    timetable = generate_timetable(class_name, section, day)
    return jsonify({"timetable": timetable.to_dict(orient="records")})

@app.route('/adjust', methods=['POST'])
def adjust():
    req = request.get_json()
    if not req or "timetable" not in req or "unavailable_teacher" not in req:
        return jsonify({"error": "Missing timetable or unavailable_teacher"}), 400
    timetable = req["timetable"]
    unavailable_teacher = req["unavailable_teacher"]
    adjusted = adjust_timetable(timetable, unavailable_teacher)
    return jsonify({"timetable": adjusted.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(debug=True, port=5000)