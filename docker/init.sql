-- Drop existing tables if they exist
DROP TABLE IF EXISTS hosts;
DROP TABLE IF EXISTS listings;

-- Create hosts table
CREATE TABLE hosts (
    host_id SERIAL PRIMARY KEY,
    host_name VARCHAR(255),
    host_response_rate INTEGER,
    host_verifications TEXT,
    is_superhost BOOLEAN,
    response_time VARCHAR(255)
);
-- Create listings table
CREATE TABLE listings (
    listing_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    city VARCHAR(255),
    property_type VARCHAR(255),
    amenities TEXT,
    price DECIMAL,
    minimum_nights INTEGER,
    review_scores_rating DECIMAL,
    instant_bookable BOOLEAN,
    host_id INTEGER REFERENCES hosts(host_id)
);
-- Copy data from local CSV files
COPY hosts(host_id, host_name, host_response_rate, host_verifications, is_superhost, response_time)
FROM '/data_folder_inside_docker_container/hosts.csv'
WITH (FORMAT csv, HEADER true);
COPY listings(listing_id, name, city, property_type, amenities, price, minimum_nights, review_scores_rating, instant_bookable, host_id)
FROM '/data_folder_inside_docker_container/listings.csv'
WITH (FORMAT csv, HEADER true);