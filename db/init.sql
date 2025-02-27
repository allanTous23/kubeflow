CREATE DATABASE IF NOT EXISTS predictions;

USE predictions;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    input_data TEXT ,
    output_data TEXT 
);
