## Dataset Description

## Background

> TODO: paste the official description from the Kaggle competition's Data tab (the page needs a login, so it couldn't be fetched).

Each row describes one vehicle by 20 numeric specifications: engine, dimensions, weight, fuel efficiency, performance, and so on. The goal is to predict the vehicle's class (`vehicle_class`, one of 3 labels: 0, 1, 2) from these specifications.

## Data Description

The dataset folder contains the following files:

- **train.csv:** 10000 x 22
- **test.csv:** 2500 x 21
- **sample_submission.csv:** 2500 x 2 (`vehicle_id,vehicle_class`)

Files were renamed from the Kaggle download names (`vehicle_train_10000.csv`, `vehicle_test.csv`, `vehicle_sample_submission.csv`) to match `dataset/regression/`.

### Columns

| Column Name                    | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `vehicle_id`                   | Unique identifier (train 1–10000, test 10001–12500). |
| `engine_displacement_l`        | Engine displacement in litres.                    |
| `horsepower_hp`                | Engine power in horsepower.                       |
| `vehicle_weight_kg`            | Vehicle weight in kg.                             |
| `wheelbase_mm`                 | Wheelbase in mm.                                  |
| `vehicle_length_mm`            | Overall length in mm.                             |
| `vehicle_width_mm`             | Overall width in mm.                              |
| `vehicle_height_mm`            | Overall height in mm.                             |
| `fuel_tank_capacity_l`         | Fuel tank capacity in litres.                     |
| `city_fuel_efficiency_kmpl`    | City fuel efficiency in km/l.                     |
| `highway_fuel_efficiency_kmpl` | Highway fuel efficiency in km/l.                  |
| `engine_cylinders`             | Number of engine cylinders (integer, 3–10).       |
| `torque_nm`                    | Engine torque in Nm.                              |
| `ground_clearance_mm`          | Ground clearance in mm.                           |
| `turning_radius_m`             | Turning radius in metres.                         |
| `cargo_capacity_l`             | Cargo capacity in litres.                         |
| `acceleration_0_100_kmh_s`     | 0–100 km/h acceleration time in seconds.          |
| `braking_distance_m`           | Braking distance in metres.                       |
| `tire_width_mm`                | Tyre width in mm.                                 |
| `drag_coefficient`             | Aerodynamic drag coefficient.                     |
| `battery_capacity_kwh`         | Battery capacity in kWh.                          |
| `vehicle_class`                | **Target.** Vehicle class label: 0, 1 or 2 (train only). |
