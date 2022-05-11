## How to use the data

TBD, import the sample data or generate your own with this project to Nebula Graph with nebula-importer.

## How to generate data

Install Python and Julia and then install dependencies:

```bash
# python dependencies
python3 -m pip install -r requirements.txt

# julia dependencies
julia install.ji

# install ABCDGraphGenerator.jl
git clone https://github.com/bkamins/ABCDGraphGenerator.jl.git
```

Configure the `config.toml` as you wish, and run:

```bash
python3 data_generator.py
```



## Graph Model

tags(vertex label)

- contact
  - properties: name, gender, birthday,
- device
- loan_applicant
  - properties: address, degree, occupation, salary, is_risky, risk_profile
- loan_application
  - properties: apply_agent_id, apply_date, application_id, approval_status, application_type, rejection_reason
- phone_number
  - properties: phone_num
- corporation
  - properties: name, is_risky, risk_profile

edge types

- with_phone_num()
- applied_for_loan(start_time)
- used_device(start_time)
- worked_for(start_time)
- is_related_to(degree)

![fraud_detection_graph_model](images/fraud_detection_graph_model.svg)

## Data Generation Process

We will be leveraging [py-Faker](https://github.com/joke2k/faker) to generate relatively reasonable properties, and [ABCDGraphGenerator.jl](https://github.com/bkamins/ABCDGraphGenerator.jl) to generate relationships with defined community structure.

As the relationship should be typed differently, i.e, `shared_phone`, `shared_employer`, `shared_device`, etc, the generation process would be as the following digram.

![fraud_detection_data_gen_process](images/fraud_detection_data_gen_process.svg)

The steps are:

0. Generate contacts(person) as vertices

1. Generate relationships/edges with configurable factors(degrees, community size, etc.)

2. Distribute relationships to different patterns:

   `shared a phone number`, `shared an employer`,  `shared a device`, `with a phone number of one's employer`, `is a relative of`

3. Generate Loan Applications, thus adding contact(person) a vertex tag of `Loan Applicant` and an `applied_for_loan` edge from the given person to the `Loan Application`.