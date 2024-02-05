from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpContinuous
import pandas as pd
import time
import os
import timeit


# Record the start time
travel_time = {
    ("Filmhaus", "Filmhaus"): 0,
    ("Filmhaus", "InfinityDome"): 3,
    ("Filmhaus", "Kolosseum"): 6,
    ("Filmhaus", "Koki"): 2,
    ("Filmhaus", "Stadthalle"): 4,
    ("InfinityDome", "Filmhaus"): 3,
    ("InfinityDome", "InfinityDome"): 0,
    ("InfinityDome", "Kolosseum"): 4,
    ("InfinityDome", "Koki"): 2,
    ("InfinityDome", "Stadthalle"): 1,
    ("Kolosseum", "Filmhaus"): 7,
    ("Kolosseum", "InfinityDome"): 4,
    ("Kolosseum", "Kolosseum"): 0,
    ("Kolosseum", "Koki"): 8,
    ("Kolosseum", "Stadthalle"): 2,
    ("Koki", "Filmhaus"): 4,
    ("Koki", "InfinityDome"): 3,
    ("Koki", "Kolosseum"): 7,
    ("Koki", "Koki"): 0,
    ("Koki", "Stadthalle"): 4,
    ("Stadthalle", "Filmhaus"): 4,
    ("Stadthalle", "InfinityDome"): 2,
    ("Stadthalle", "Kolosseum"): 2,
    ("Stadthalle", "Koki"): 5,
    ("Stadthalle", "Stadthalle"): 0
}


def overlap(showing1, showing2, df):
    film1, date1, time1 = showing1.split('_')
    film2, date2, time2 = showing2.split('_')
    if date1 != date2:
        return False
    start1, end1, venue1 = get_times_and_location(film1, time1, df)
    start2, end2, venue2 = get_times_and_location(film2, time2, df)

    if venue_differ(showing1, showing2, df):
        end1 += travel_time[(venue1, venue2)]
    return (start1 < end2 and start2 + 1 < end1)

def get_times_and_location(film_name, time_str, df):
    film_data = df[(df[0] == film_name) & (df[2] == time_str)].iloc[0]
    start_time = convert_to_minutes(film_data[2])
    duration = int(film_data[3])  # Make sure this is an integer
    end_time = start_time + duration
    venue = film_data[4]
    if "CineStar" in venue:
        venue = "Stadthalle"
    if "Filmhaus" in venue:
        venue = "Filmhaus"
    return start_time, end_time, venue

def convert_to_minutes(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

def venue_differ(showing1, showing2, df):
    film1, date1, time1 = showing1.split('_')
    film2, date2, time2 = showing2.split('_')
    _, _, venue1 = get_times_and_location(film1, time1,df )
    _, _, venue2 = get_times_and_location(film2, time2, df)
    return venue1 != venue2


def process_dataset(dataset_path):
    start_time = time.time()
    df = pd.read_csv(dataset_path, sep=";", header=None)

    films = df[0].tolist()
    showings = [f"{row[0]}_{row[1]}_{row[2]}" for _, row in df.iterrows()]
    rating = dict(zip(showings, df.iloc[:, 5]))

    film_data = {}
    for _, row in df.iterrows():
        unique_key = f"{row[0]}_{row[1]}_{row[2]}"  # Remove row[4] from the key format
        film_data[unique_key] = {
            'FilmName': row[0],
            'Time': row[2],
            'Duration': int(row[3]),
            'Venue': row[4],
            'Rating': row[5]
        }

    # Create LP variables
    unique_films = set(df[0].tolist())
    film_vars = LpVariable.dicts("Film", unique_films, 0, 1, LpBinary)
    showing_indices = range(len(showings))
    showing_vars = LpVariable.dicts("Showing", showing_indices, 0, 1, LpBinary)

    # Create LP problem
    prob = LpProblem("Film Festival Schedule", LpMaximize)

    # Objective function
    objective_expr = lpSum([film_data[showings[i]]['Rating'] * showing_vars[i] for i in showing_indices])
    prob += objective_expr, "Total Rating"

    # Constraints for overlapping showings
    for i in showing_indices:
        for j in showing_indices:
            if i < j and overlap(showings[i], showings[j], df):
                prob += showing_vars[i] + showing_vars[j] <= 1

    # Constraints to ensure each film is watched only once
    for film in unique_films:
        showing_indices_for_film = [i for i in showing_indices if film in showings[i]]
        prob += lpSum([showing_vars[i] for i in showing_indices_for_film]) <= 1, f"Watch_{film}_Once"

        for i in showing_indices_for_film:
            for j in showing_indices_for_film:
                if i < j and overlap(showings[i], showings[j], df):
                    prob += showing_vars[i] + showing_vars[j] <= 1

    # Save the LP as a file
    prob.writeLP("film_schedule.lp")
    print(prob)
    # Solve the problem
    prob.solve()

    # Print selected showings
    selected_showings = [showings[i] for i in showing_indices if showing_vars[i].varValue > 0]
    for showing in selected_showings:
        print(f"Watch showing: {showing}")

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Script executed in {runtime:.2f} seconds")
    objective_value = prob.objective.value()

    return objective_value, runtime

def process_datasets(dataset_directory, output_file):
    with open(output_file, 'w') as outfile:
        outfile.write("Dataset, Objective Value, Runtime (seconds)\n")

        for filename in os.listdir(dataset_directory):
            if filename.endswith(".csv"):
                dataset_path = os.path.join(dataset_directory, filename)
                objective_value, runtime = process_dataset(dataset_path)  # Pass the output_file

                # Write dataset info to the output file
                outfile.write(f"{filename}, {objective_value}, {runtime}\n")

dataset_directory = "small datasets"
output_file = "small_objective_values.csv"

process_datasets(dataset_directory, output_file)

