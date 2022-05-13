## How to use the data

TBD, import the sample data or generate your own with this project to Nebula Graph with nebula-importer.

## How to generate data

Install Python3 and Julia and then install dependencies with:

```bash
# python dependencies
python3 -m pip install -r requirements.txt

# julia dependencies
julia install.ji
```

Configure the `config.toml` as you wish, where options were documented inline, then just run:

```bash
python3 data_generator.py
```

Data will be output under the `data` folder, the files under `data_sample` could be used if it fits your needs. The process should looks like:

https://user-images.githubusercontent.com/1651790/168299297-83b232a1-23b4-44e0-b569-595b70a2b0da.mp4

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

As the relationship should be typed differently, i.e, `shared_phone`, `shared_employer`, `shared_device`, etc, the generation process would be as the following diagram.

![fraud_detection_data_gen_process](images/fraud_detection_data_gen_process.svg)

The steps are:

0. Generate contacts(person) as vertices

1. Generate relationships/edges with configurable factors(degrees, community size, etc.)

2. Distribute relationships to different patterns:

   `shared a phone number`, `shared an employer`,  `shared a device`, `with a phone number of one's employer`, `is a relative of`

3. Generate Loan Applications, thus adding contact(person) a vertex tag of `Loan Applicant` and an `applied_for_loan` edge from the given person to the `Loan Application`.



## Data Explanation

```bash
$tree data
data
├── abcd            # raw data with ABCD Sampler, reference only
│   ├── com.dat     # vertex -> community
│   ├── cs.dat      # community size
│   ├── deg.dat     # vertex degree
│   └── edge.dat    # edges(which construct the community)
├── applicant_application_relationship.csv
│                   # app vertex and person-applied-> app edge
├── corporation.csv # corporation vertex
├── device.csv      # device vertex
├── is_relative_relationship.csv
│                   # is_relative (:p)-[:is_related_to]->(:p)
├── person.csv      # contact vertex
├── phone_num.csv   # phone number vertex
├── shared_device_relationship.csv
│                   # (:p)-[:used_dev]->(:dev)<-[:used_dev]-(:p)
├── shared_employer_relationship.csv
│                   # (:p)-[:worked_for]->(:corp)<-[:worked_for]-(:p)
├── shared_phone_num_relationship.csv
│                   # (src:person) -[:is_related_to]->(dst:person)
└── shared_via_employer_phone_num_relationship.csv
    # (:p)-[:worked_for]->(:corp)->(:phone_num)<-[:with_phone_num]-(:p)
```

