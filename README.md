## How to use the data

First, to bootstrap your Nebula Graph cluster, here, for a single server playground, try with [nebula-up](https://github.com/wey-gu/nebula-up/).

Then, assuming you have a Nebula Graph cluster running in docker with network namespace: `nebula-net`, you can use the following command call Nebula Graph Importer to import data into Nebula Graph, with its configuration from `nebula_graph_importer.yaml`:

```bash
# If we are using the sample data:
# cp -r data_sample_numerical_vertex_id data

# only do this for once, remove header line from data/*.csv
sed -i '1d' data/*.csv

docker run --rm -ti \
    --network=nebula-net \
    -v ${PWD}:/root/ \
    -v ${PWD}/data/:/data \
    vesoft/nebula-importer:v3.1.0 \
    --config /root/nebula_graph_importer.yaml
```

> Note, to leverage the data in [NebulaGraph Algorithm](https://github.com/vesoft-inc/nebula-algorithm/), it's recommended to configure `vertex_id_format` as `numerical`.
> This is an example to run the Louvain algorithm:
```bash
cd ~/.nebula-up/nebula-up/spark

docker exec -it sparkmaster /spark/bin/spark-submit \
    --master "local" --conf spark.rpc.askTimeout=6000s \
    --class com.vesoft.nebula.algorithm.Main \
    --driver-memory 4g /root/download/nebula-algo.jar \
    -p /root/louvain.conf
```

## How to generate data

Install Python3 and [Julia](https://www.google.com/search?q=how+to+install+julia) first, then install dependencies with:

```bash
# python dependencies
python3 -m pip install -r requirements.txt

# julia dependencies, please install julia before running this line
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
  - properties: name, gender, birthday
- device
- loan_applicant
  - properties: address, degree, occupation, salary, is_risky, risk_profile, name, gender, birthday
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

See comments inline, i.e., for `shared_via_employer_phone_num_relationship.csv`, its comment is:
```cypher
(:p)-[:worked_for]->(:corp)-[:with_phone_num]->(:phone_num)<-[:with_phone_num]-(:p)
```
This means one line of records in the CSV file contains three edges and four vertices:
- `:p` is a person vertex tag
- `worked_for` is a edge type between `:p` and `:corp`
- `:corp` is a corporation vertex tag
- `with_phone_num` is a edge type between corporation and phone_num
- `:phone_num` is a phone number vertex tag
- `with_phone_num` is a edge type between phone_num and person
- `:p` is another person vertex tag


```bash
$tree data
data
├── abcd             # raw data with ABCD Sampler, reference only
│   ├── com.dat      # vertex -> community
│   ├── cs.dat       # community size
│   ├── deg.dat      # vertex degree
│   └── edge.dat     # edges(which construct the community)
├── applicant_application_with_is_related_to.csv
│                    # (loan_applicant:appliant)-[:is_related_to]->(contact:person)
│                    # (loan_applicant:appliant)-[:applied_for_loan]->(app:loan_application)
├── applicant_application_with_shared_device.csv
│                    # (loan_applicant_0:appliant)-[:used_dev]->(:dev)<-[:used_dev]-(loan_applicant_1:appliant)
│                    # (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
│                    # (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
├── applicant_application_with_shared_phone_num.csv
│                    # (loan_applicant_0:appliant)-[:with_phone_num]->(:phone_num)<-[:with_phone_num]-(loan_applicant_1:appliant)
│                    # (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
│                    # (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
├── applicant_application_with_shared_employer.csv
│                    # (loan_applicant_0:appliant)-[:worked_for]->(:corp)<-[:worked_for]-(loan_applicant_1:appliant)
│                    # (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
│                    # (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
├── applicant_application_connected_with_employer_and_phone_num.csv
│                    # (loan_applicant_0:appliant)-[:worked_for]->(:corp)-[:with_phone_num]->(:phone_num)<-[:with_phone_num]-(loan_applicant_1:appliant)
│                    # (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
│                    # (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
├── corporation.csv  # corporation vertex
├── device.csv       # device vertex
├── is_relative_relationship.csv
│                    # is_relative (:p)-[:is_related_to]->(:p)
├── person.csv       # contact vertex
├── phone_number.csv # phone number vertex
├── shared_device_relationship.csv
│                    # (:p)-[:used_dev]->(:dev)<-[:used_dev]-(:p)
├── shared_employer_relationship.csv
│                    # (:p)-[:worked_for]->(:corp)<-[:worked_for]-(:p)
├── shared_phone_num_relationship.csv
│                    # (:p)-[:with_phone_num]->(:phone_num)<-[:with_phone_num]-(:p)
└── shared_via_employer_phone_num_relationship.csv
    # (:p)-[:worked_for]->(:corp)-[:with_phone_num]->(:phone_num)<-[:with_phone_num]-(:p)
```

