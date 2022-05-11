import csv
import julia
import os
import re
import toml

from faker import Faker
from pathlib import Path
from random import randint
import numpy as np


CONF_PATH = "config.toml"
ABCD_SAMPLER_PATH = "abcd_sampler.jl"
WRITE_BATCH = 1000


class FraudDetectionDataGenerator:
    """
    Generate data.
    """

    def __init__(self, conf_path=CONF_PATH):
        self.conf_path = conf_path
        self.conf = self.get_conf()
        self.faker = Faker(self.conf["data_language"])
        self.person_count = int(self.conf["n"])
        self.person_id_prefix = self.conf["person_id_prefix"]
        self.abcd_edge_path = self.conf["networkfile"]
        self.abcd_data_dir = os.path.dirname(self.abcd_edge_path)
        self.phone_number_count = int(self.conf["phone_number_count"])
        self.phone_number_id_prefix = self.conf["phone_number_id_prefix"]
        self.device_id_prefix = self.conf["device_id_prefix"]
        self.device_count = int(self.conf["device_count"])
        self.corporation_count = int(self.conf["corporation_count"])
        self.loan_application_count = int(self.conf["loan_application_count"])


    def get_conf(self):
        with open(self.conf_path) as conf_file:
            conf = conf_file.read()
            return toml.loads(conf)

    def init_julia(self):
        """
        Initialize Julia environment
        """
        julia.install()
        return julia.Julia()

    @staticmethod
    def run_abcd_sample(julia, data_path):
        """
        Run ABCD sample.

        julia: julia environment
        data_path: path to data
        """
        # Create data directory if not exists
        Path(data_path).mkdir(parents=True, exist_ok=True)
        # Run ABCD sample to generate Relationship data with community structure
        julia.include(ABCD_SAMPLER_PATH)

    @staticmethod
    def csv_writer(file_path,
                   row_count,
                   row_generator,
                   index=False,
                   index_prefix="",
                   header=None):
        """
        Write rows to csv file.

        file_path: path to csv file
        row_count: number of rows to write
        row_generator: generator to generate rows
        index: whether to add index, started from 1 to be consistent with ABCD sample
        index_prefix: prefix of index
        header: header of csv file
        """
        with open(file_path, mode="w") as file:
            if index:
                cursor = 1
            writer = csv.writer(file,
                                delimiter=",",
                                quotechar="'",
                                quoting=csv.QUOTE_MINIMAL)
            csv_buffer = list()
            if header:
                writer.writerow(header)
            for _ in range(row_count):
                if index:
                    csv_buffer.append((f"{index_prefix}{cursor}", ) +
                                      row_generator())
                    cursor += 1
                else:
                    csv_buffer.append(row_generator())
                if len(csv_buffer) > WRITE_BATCH:
                    writer.writerows(csv_buffer)
                    del csv_buffer[:]
            if csv_buffer:
                writer.writerows(csv_buffer)
                del csv_buffer[:]

    # contact generator
    def person_generator(self):
        """
        property: (name, gender, birthday)
        """
        return (self.faker.name(),
                "M" if self.faker.boolean() else "F",
                self.faker.date_of_birth())


    def phone_generator(self):
        """
        properties: (phone_number)
        """
        return (self.faker.phone_number(),)

    def device_generator(self):
        """
        properties: (device_id)
        """
        return (self.faker.md5(),)

    def corporation_generator(self):
        """
        properties: (name, address, is_risky, risk_profile)
        """
        is_risky = self.faker.boolean(chance_of_getting_true=self.conf["chance_of_corporation_risky_percentage"])
        risk_profile = "" if not is_risky else self.faker.sentence()
        return (self.faker.company(),
                self.faker.address(),
                is_risky,
                risk_profile)

    def loan_applicant_generator(self):
        """
        properties: (address, degree, occupation, salary, is_risky, risk_profile)
        """
        is_risky = self.faker.boolean(chance_of_getting_true=self.conf["chance_of_applicant_risky_percentage"])
        risk_profile = "" if not is_risky else self.faker.sentence()
        return(self.faker.address(),
                self.faker.random_element(elements=["Bachelor", "Master", "PhD"]),
                self.faker.job(),
                self.faker.random_number(digits=6),
                is_risky,
                risk_profile)

    def loan_application_generator(self):
        """
        properties: (apply_agent_id, apply_date, application_id, approval_status, application_type, rejection_reason)
        """
        return (self.faker.md5(),
                self.faker.date_between(start_date="-30y", end_date="today"),
                self.faker.md5(),
                self.faker.random_element(elements=["Approved", "Rejected"]),
                self.faker.random_element(elements=["Loan", "Credit"]),
                self.faker.sentence())

    def loan_applicant_and_application_generator(self):
        """
        (src:loan_applicant:person) -[:applied_for_loan]->(app:loan_application)
        """
        appliant_id = self.person_id_prefix + str(self.faker.random_number % self.person_count)
        appliant = self.loan_applicant_generator()
        application = self.loan_application_generator()
        start_date = application[1]
        return (appliant_id,) + appliant + application + (start_date,)


    def shared_phone_number_relationship_generator(self):
        """
        relationship pattern:
            (src:person) -[:with_phone_num]->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        CSV fields:
            src_id, dst_id, pn_id
        Note, src_id, dst_id comes from abcd_sampler.jl, we just randomly generate a pn_id
        """
        return (self.faker.random_number() % self.phone_number_count,)


    def shared_device_relationship_generator(self):
        """
        relationship pattern:
            (src:person) -[:used_device]->(dev:device)<-[:used_device]-(dst:person)
        CSV fields:
            src_id, dst_id, device_id, src_device_start_time, dst_device_start_time
        Note, src_id, dst_id comes from abcd_sampler.jl, we just randomly generate:
            device_id, src_device_start_time, dst_device_start_time
        """
        return (self.faker.random_number() % self.device_count,
                self.faker.date_between(start_date="-3y", end_date="today"),
                self.faker.date_between(start_date="-3y", end_date="today"))

    def shared_employer_relationship_generator(self):
        """
        relationship pattern:
            (src:person) -[:worked_for]->(corp:corporation)<-[:worked_for]-(dst:person)
        CSV fields:
            src_id, dst_id, corp_id, src_work_for_start_time, dst_work_for_start_time
        Note, src_id, dst_id comes from abcd_sampler.jl, we just randomly generate:
            corp_id, src_work_for_start_time, dst_work_for_start_time
        """
        return (self.faker.random_number() % self.corporation_count,
                self.faker.date_between(start_date="-3y", end_date="today"),
                self.faker.date_between(start_date="-3y", end_date="today"))

    def via_employer_phone_number_relationship_generator(self):
        """
        relationship pattern:
            (src:person) -[:worked_for]->(corp:corporation)->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        CSV fields:
            src_id, dst_id, corp_id, pn_id, src_work_for_start_time
        Note, src_id, dst_id comes from abcd_sampler.jl, we just randomly generate:
            corp_id, pn_id, src_work_for_start_time
        """
        return (self.faker.random_number() % self.corporation_count,
                self.faker.random_number() % self.phone_num_count,
                self.faker.date_between(start_date="-3y", end_date="today"))

    def is_related_to_relationship_generator(self):
        """
        relationship pattern:
            (src:person) -[:is_related_to]->(dst:person)
        CSV fields:
            src_id, dst_id, degree
        Note, src_id, dst_id comes from abcd_sampler.jl, we just randomly generate:
            degree
        """
        return (self.faker.random_number() % 100,)    


    def generate_contacts(self):
        """
        Generate contacts.
        """
        self.csv_writer("data/person.csv",
                        self.person_count,
                        self.person_generator,
                        index=True,
                        index_prefix=self.person_id_prefix)

    def generate_phones_numbers(self):
        """
        Generate phones numbers.
        """
        self.csv_writer("data/phone_num.csv",
                        self.phone_number_count,
                        self.phone_generator,
                        index=True,
                        index_prefix=self.phone_number_id_prefix)

    def generate_devices(self):
        """
        Generate devices.
        """
        self.csv_writer("data/device.csv",
                        self.device_count,
                        self.device_generator,
                        index=True,
                        index_prefix=self.device_id_prefix)

    def generate_clusterred_contacts_relations(self):
        """
        Generate clusterred contacts relations.
        """
        self.generate_phones_numbers()
        self.generate_devices()
        edges = np.loadtxt(self.abcd_edge_path, delimiter=",")
        edge_count = edges.shape[0]

        # Get count of different patterns of relationship
        shared_phone_num_count = int(float(self.conf["relation_via_phone_num_ratio"]) * edge_count)
        shared_device_count = int(float(self.conf["relation_via_device_ratio"]) * edge_count)
        shared_employer_count = int(float(self.conf["relation_shared_employer_ratio"]) * edge_count)
        shared_via_employer_phone_num_count = int(float(self.conf["relation_via_employer_phone_num_ratio"]) * edge_count)
        is_relative_count = edge_count - sum((
            shared_phone_num_count, shared_device_count, shared_employer_count, shared_via_employer_phone_num_count))

        # (src:person) -[:with_phone_num]->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        # write intermediate data to be concatenated to the final csv file
        self.csv_writer("data/_shared_phone_num_relationship.csv",
                        shared_phone_num_count,
                        self.shared_phone_number_relationship_generator,
                        index=False)
        shared_num_rels = np.loadtxt("data/_shared_phone_num_relationship.csv", delimiter=",")
        start_index = 0,
        end_index = shared_phone_num_count
        concat_shared_num_rels = np.concatenate((edges[0:shared_phone_num_count], shared_num_rels), axis=1)
        np.savetxt("data/shared_phone_num_relationship.csv", concat_shared_num_rels, delimiter=",",
                   header="src_person_id, dst_person_id, phone_num_id")
        
        # (src:person) -[:used_device]->(d:device)<-[:used_device]-(dst:person)
        # write intermediate data to be concatenated to the final csv file
        self.csv_writer("data/_shared_device_relationship.csv",
                        shared_device_count,
                        self.shared_device_relationship_generator,
                        index=False)
        shared_device_rels = np.loadtxt("data/_shared_device_relationship.csv", delimiter=",")
        start_index = shared_phone_num_count,
        end_index = shared_phone_num_count + shared_device_count
        concat_shared_device_rels = np.concatenate((edges[start_index:end_index], shared_device_rels), axis=1)
        np.savetxt("data/shared_device_relationship.csv", concat_shared_device_rels, delimiter=",",
                   header="src_person_id, dst_person_id, device_id, src_device_start_time, dst_device_start_time")
        
        # (src:person) -[:worked_for]->(corp:corporation)<-[:worked_for]-(dst:person)
        # write intermediate data to be concatenated to the final csv file
        self.csv_writer("data/_shared_employer_relationship.csv",
                        shared_employer_count,
                        self.shared_employer_relationship_generator,
                        index=False)
        shared_employer_rels = np.loadtxt("data/_shared_employer_relationship.csv", delimiter=",")
        start_index = shared_phone_num_count + shared_device_count,
        end_index = shared_phone_num_count + shared_device_count + shared_employer_count
        concat_shared_employer_rels = np.concatenate((edges[start_index:end_index], shared_employer_rels), axis=1)
        np.savetxt("data/shared_employer_relationship.csv", concat_shared_employer_rels, delimiter=",",
                     header="src_person_id, dst_person_id, corp_id, src_work_for_start_time, dst_work_for_start_time")
        
        # (src:person) -[:worked_for]->(corp:corporation)->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        # write intermediate data to be concatenated to the final csv file
        self.csv_writer("data/_shared_via_employer_phone_num_relationship.csv",
                        shared_via_employer_phone_num_count,
                        self.via_employer_phone_number_relationship_generator,
                        index=False)
        shared_via_employer_phone_num_rels = np.loadtxt("data/_shared_via_employer_phone_num_relationship.csv", delimiter=",")
        start_index = shared_phone_num_count + shared_device_count + shared_employer_count
        end_index = shared_phone_num_count + shared_device_count + shared_employer_count + shared_via_employer_phone_num_count
        concat_shared_via_employer_phone_num_rels = np.concatenate((edges[start_index:end_index], shared_via_employer_phone_num_rels), axis=1)
        np.savetxt("data/shared_via_employer_phone_num_relationship.csv", concat_shared_via_employer_phone_num_rels, delimiter=",",
                     header="src_person_id, dst_person_id, corp_id, phone_num_id, src_work_for_start_time")
        
        # (src:person) -[:is_related_to]->(dst:person)
        # write intermediate data to be concatenated to the final csv file
        self.csv_writer("data/_is_relative_relationship.csv",
                        is_relative_count,
                        self.is_related_to_relationship_generator,
                        index=False)
        is_relative_rels = np.loadtxt("data/_is_relative_relationship.csv", delimiter=",")
        start_index = shared_phone_num_count + shared_device_count + shared_employer_count + shared_via_employer_phone_num_count
        end_index = edge_count
        concat_is_relative_rels = np.concatenate((edges[start_index:end_index], is_relative_rels), axis=1)
        np.savetxt("data/is_relative_relationship.csv", concat_is_relative_rels, delimiter=",",
                     header="src_person_id, dst_person_id, src_is_relative_start_time, dst_is_relative_start_time")

    def generate_applicants_and_applications(self):
        # (src:loan_applicant:person) -[:applied_for_loan]->(app:loan_application)
        header = ["loan_application_id", "appliant_person_id",
            "address", "degree", "occupation", "salary", "is_risky", "risk_profile",
            "apply_agent_id", "apply_date", "application_id", "approval_status", "application_type", "rejection_reason"]
        self.csv_writer("data/applicant_application_relationship.csv",
                        self.loan_application_count,
                        self.applicant_application_relationship_generator,
                        index=True,
                        index_prefix=self.conf["loan_application_id_prefix"],
                        header=header)

if __name__ == "__main__":

    gen = FraudDetectionDataGenerator()
    # Step 0: Generate contacts(person) as vertices
    gen.generate_contacts()
    # Step 1: Run ABCD sample to generate relationship data with community structure
    julia, data_path = gen.init_julia(), gen.abcd_data_dir
    gen.run_abcd_sample(julia, data_path)
    # Step 2: Distribute relationships to different patterns
    gen.generate_clusterred_contacts_relations()
    # Step 3: Generate applicant and applications
    gen.generate_applicants_and_applications()

