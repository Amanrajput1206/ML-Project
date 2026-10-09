# ML Project - Domestic Flight Price Prediction Challenge

read the project plan for steps and final-report-prototype for expected outcome.

## Dataset Description

### Background

Booking domestic flights involves several factors that influence ticket prices:

- Different airlines offer varying fare structures and services.
- Flight prices change based on the source and destination cities.
- Departure and arrival times can affect demand and pricing.
- The number of stops and total flight duration influence travel convenience and cost.
- Flight class, such as Economy or Business, significantly impacts the ticket price.
- Prices may vary depending on the number of days remaining before departure.

The goal is to build a model that predicts the ticket price of each domestic flight, enabling better travel planning, fare estimation, and data-driven pricing analysis.

### Data Description

The dataset folder contains the following files:

- **train.csv:** 15589 × 12
- **test.csv:** 3898 × 11
- **sample_submission.csv:** 3898 × 2

### Columns

| Column Name        | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| `id`               | Unique identifier for each flight record.                              |
| `airline`          | Name of the airline operating the flight.                              |
| `flight`           | Alphanumeric flight number or identifier.                              |
| `source_city`      | City from which the flight departs.                                    |
| `departure_time`   | Time-of-day category when the flight departs.                          |
| `stops`            | Number of stops or layovers during the journey.                        |
| `arrival_time`     | Time-of-day category when the flight arrives.                          |
| `destination_city` | City where the flight arrives.                                         |
| `flight_class`     | Seating class of the ticket, such as Economy or Business.              |
| `duration`         | Total flight duration in hours.                                        |
| `days_left`        | Number of days remaining before the flight departure date.             |
| `price`            | Target variable representing the flight ticket price in Indian Rupees. |
