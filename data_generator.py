import csv
import os
import re
import toml

from collections import OrderedDict
from faker import Faker
from multiprocessing import Pool
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.syntax import Syntax

from pathlib import Path

import pandas as pd

CONF_PATH = "config.toml"
ABCD_SAMPLER_PATH = "abcd_sampler.jl"
WRITE_BATCH = 1000

console = Console()


def log(message):
    console.print("\n[bold bright_cyan][ Info:[/bold bright_cyan]", message)


def title(title, description=None):
    table = Table(show_header=True)
    table.add_column(title, style="dim", width=96)
    if description:
        table.add_row(description)
    console.print("\n", table)


OUTPUT_EXPLANATION = """
$ tree
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
"""


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
        self.applicant_id_prefix = self.conf["applicant_id_prefix"]
        self.abcd_edge_path = self.conf["networkfile"]
        self.abcd_data_dir = os.path.dirname(self.abcd_edge_path)
        self.phone_number_count = int(self.conf["phone_number_count"])
        self.phone_number_id_prefix = self.conf["phone_number_id_prefix"]
        self.device_id_prefix = self.conf["device_id_prefix"]
        self.device_count = int(self.conf["device_count"])
        self.corporation_count = int(self.conf["corporation_count"])
        self.corporation_id_prefix = self.conf["corporation_id_prefix"]
        self.process_count = int(self.conf["process_count"])
        Path("data").mkdir(parents=True, exist_ok=True)

    def get_conf(self):
        with open(self.conf_path) as conf_file:
            conf = conf_file.read()
            return toml.loads(conf)

    def print_conf(self):
        with open(self.conf_path) as conf_file:
            conf = conf_file.read()
            toml_highlight = Syntax(conf,
                                    "toml",
                                    theme="monokai",
                                    line_numbers=False)
            title("Getting [bold magenta]config.toml[/bold magenta] parsed",
                  toml_highlight)

    def init_julia(self):
        """
        Initialize Julia environment
        """
        import julia

        julia.install()
        return julia.Julia()

    def run_abcd_sample(self, data_path):
        """
        Run ABCD sample.

        data_path: path to data
        """
        from julia import Main

        # Create data directory if not exists
        Path(data_path).mkdir(parents=True, exist_ok=True)
        # Run ABCD sample to generate Relationship data with community structure
        log(f"Calling [bold magenta]{ABCD_SAMPLER_PATH}[/bold magenta] to generate community structured data..."
            )
        Main.include(ABCD_SAMPLER_PATH)
        log(f"Calling [bold magenta]{ABCD_SAMPLER_PATH}[/bold magenta]...[green]✓[/green]. Data generated at [bold magenta]{self.abcd_data_dir}[/bold magenta]"
            )

    @staticmethod
    def csv_writer(file_path,
                   row_count,
                   row_generator,
                   index=False,
                   index_prefix="",
                   header=None,
                   init_index=1):
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
                cursor = int(init_index)
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
        return (self.faker.name(), "M" if self.faker.boolean() else "F",
                self.faker.date_of_birth())

    def phone_generator(self):
        """
        properties: (phone_number)
        """
        return (self.faker.phone_number(), )

    def device_generator(self):
        """
        properties: (device_id)
        """
        return (self.faker.md5(), )

    def corporation_generator(self):
        """
        properties: (name, address, is_risky, risk_profile)
        """
        is_risky = self.faker.boolean(chance_of_getting_true=float(
            self.conf["chance_of_corporation_risky_percentage"]))
        risk_profile = "NA" if not is_risky else self.faker.sentence()
        return (re.escape(self.faker.company().replace(", ", "_")),
                re.escape(self.faker.address().replace("\n",
                                                       " ").replace(",", " ")),
                is_risky, risk_profile)

    def loan_applicant_generator(self):
        """
        properties: (address, degree, occupation, salary, is_risky, risk_profile)
        """
        is_risky = self.faker.boolean(chance_of_getting_true=float(
            self.conf["chance_of_applicant_risky_percentage"]))
        risk_profile = "NA" if not is_risky else self.faker.sentence()
        return (re.escape(self.faker.address().replace("\n",
                                                       " ").replace(",", " ")),
                self.faker.random_element(
                    elements=["Bachelor", "Master", "PhD"]),
                re.escape(self.faker.job().replace(",", " ")),
                self.faker.random_number(digits=6), is_risky, risk_profile)

    def loan_application_generator(self):
        """
        properties: (apply_agent_id, apply_date, application_uuid, approval_status, application_type, rejection_reason)
        """
        chance_of_application_rejected = float(
            self.conf["chance_of_application_rejected"])
        approval_status = self.faker.random_element(elements=OrderedDict([(
            "Approved", 1 - chance_of_application_rejected
        ), ("Rejected", chance_of_application_rejected)]))
        if approval_status == "Rejected":
            rejection_reason = self.faker.sentence()
        else:
            rejection_reason = "NA"
        return (self.faker.md5(),
                self.faker.date_between(start_date="-30y", end_date="today"),
                self.faker.md5(), approval_status,
                self.faker.random_element(elements=["Loan", "Credit"]),
                rejection_reason)

    def loan_applicant_and_application_generator(self):
        """
        MATCH Pattern:
        (loan_applicant:appliant)-[:applied_for_loan]->(app:loan_application)
        appliant_id comes from "p_*" records
        """
        appliant = self.loan_applicant_generator()
        application = self.loan_application_generator()
        start_date = application[1]
        return appliant + application + (start_date, )

    def shared_phone_number_relationship_generator(self):
        """
        relationship pattern:
            (src:person) -[:with_phone_num]->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        CSV fields:
            src_id, dst_id, pn_id
        Note, src_id, dst_id comes from abcd_sampler.jl, we just randomly generate a pn_id
        """
        return (
            f"{self.phone_number_id_prefix}{self.faker.random_number() % self.phone_number_count + 1}",
        )

    def shared_device_relationship_generator(self):
        """
        relationship pattern:
            (src:person) -[:used_device]->(dev:device)<-[:used_device]-(dst:person)
        CSV fields:
            src_id, dst_id, device_id, src_device_start_time, dst_device_start_time
        Note, src_id, dst_id comes from abcd_sampler.jl, we just randomly generate:
            device_id, src_device_start_time, dst_device_start_time
        """
        return (
            f"{self.device_id_prefix}{self.faker.random_number() % self.device_count + 1}",
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
        return (
            f"{self.corporation_id_prefix}{self.faker.random_number() % self.corporation_count + 1}",
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
        return (
            f"{self.corporation_id_prefix}{self.faker.random_number() % self.corporation_count + 1}",
            f"{self.phone_number_id_prefix}{self.faker.random_number() % self.phone_number_count + 1}",
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
        return (self.faker.random_number() % 100, )

    def generate_contacts(self):
        """
        Generate contacts.
        """
        log("Generating contacts...")
        path = "data/person.csv"
        header = ["person_id", "name", "gender", "birthday"]
        self.csv_writer(path,
                        self.person_count,
                        self.person_generator,
                        index=True,
                        index_prefix=self.person_id_prefix,
                        header=header)
        log(f"Generating contacts...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]"
            )

    def generate_phones_numbers(self):
        """
        Generate phones numbers.
        """
        log("Generating phones numbers...")
        path = "data/phone_number.csv"
        header = ["phone_number_id", "phone_number"]
        self.csv_writer(path,
                        self.phone_number_count,
                        self.phone_generator,
                        index=True,
                        index_prefix=self.phone_number_id_prefix,
                        header=header)
        log(f"Generating phones numbers...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]"
            )

    def generate_devices(self):
        """
        Generate devices.
        """
        log("Generating devices...")
        path = "data/device.csv"
        header = ["device_id", "device_name"]
        self.csv_writer(path,
                        self.device_count,
                        self.device_generator,
                        index=True,
                        index_prefix=self.device_id_prefix,
                        header=header)
        log(f"Generating devices...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]"
            )

    def generate_corporations(self):
        """
        Generate corporations.
        """
        log("Generating corporations...")
        path = "data/corporation.csv"
        header = [
            "corp_id", "corp_name", "corp_address", "is_risky", "risk_profile"
        ]
        self.csv_writer(path,
                        self.corporation_count,
                        self.corporation_generator,
                        index=True,
                        index_prefix=self.corporation_id_prefix,
                        header=header)
        log(f"Generating corporations...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]"
            )

    def generate_clusterred_contacts_relations(self):
        """
        Generate clusterred contacts relations.
        """
        log("Generating clusterred contacts/person relations...")
        self.generate_phones_numbers()
        self.generate_devices()
        self.generate_corporations()
        edges = pd.read_csv(self.abcd_edge_path, delimiter=",", header=None)
        edge_count = edges.shape[0]

        # Get count of different patterns of relationship
        shared_phone_num_count = int(
            float(self.conf["relation_via_phone_num_ratio"]) * edge_count)
        shared_device_count = int(
            float(self.conf["relation_via_device_ratio"]) * edge_count)
        shared_employer_count = int(
            float(self.conf["relation_shared_employer_ratio"]) * edge_count)
        shared_via_employer_phone_num_count = int(
            float(self.conf["relation_via_employer_phone_num_ratio"]) *
            edge_count)
        is_relative_count = edge_count - sum(
            (shared_phone_num_count, shared_device_count,
             shared_employer_count, shared_via_employer_phone_num_count))

        # (src:person)-[:with_phone_num]->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        _ = "(src:person)-[:with_phone_num]->(pn:phone_number)<-[:with_phone_num]-(dst:person)"
        log("Generating shared phone number relationships in pattern:")
        title("shared a phone number", Syntax(_, "cypher", line_numbers=False))
        # write intermediate data to be column_stacked to the final csv file
        self.csv_writer("data/_shared_phone_num_relationship.csv",
                        shared_phone_num_count,
                        self.shared_phone_number_relationship_generator,
                        index=False)
        shared_num_rels = pd.read_csv(
            "data/_shared_phone_num_relationship.csv",
            delimiter=",",
            header=None)
        start_index = 0,
        end_index = shared_phone_num_count
        concat_shared_num_rels = pd.concat(
            (edges[0:shared_phone_num_count], shared_num_rels), axis=1)
        # header "src_person_id, dst_person_id, phone_num_id"
        header = ["src_id", "dst_id", "pn_id"]
        _path = "data/shared_phone_num_relationship.csv"
        concat_shared_num_rels.to_csv(_path,
                                      sep=",",
                                      index=False,
                                      header=header)
        os.remove("data/_shared_phone_num_relationship.csv")
        log(f"Generating shared phone number relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]"
            )

        # (src:person)-[:used_device]->(d:device)<-[:used_device]-(dst:person)
        _ = "(src:person)-[:used_device]->(d:device)<-[:used_device]-(dst:person)"
        log("Generating shared device relationships in pattern:")
        title("shared a device", Syntax(_, "cypher", line_numbers=False))
        # write intermediate data to be column_stacked to the final csv file
        self.csv_writer("data/_shared_device_relationship.csv",
                        shared_device_count,
                        self.shared_device_relationship_generator,
                        index=False)
        shared_device_rels = pd.read_csv(
            "data/_shared_device_relationship.csv", delimiter=",", header=None)
        start_index = shared_phone_num_count
        end_index = shared_phone_num_count + shared_device_count
        concat_shared_device_rels = pd.concat(
            (edges[start_index:end_index].reset_index(drop=True),
             shared_device_rels),
            axis=1)
        # header "src_person_id, dst_person_id, device_id, src_device_start_time, dst_device_start_time"
        header = [
            "src_id", "dst_id", "d_id", "src_device_start_time",
            "dst_device_start_time"
        ]
        _path = "data/shared_device_relationship.csv"
        concat_shared_device_rels.to_csv(_path,
                                         sep=",",
                                         index=False,
                                         header=header)
        os.remove("data/_shared_device_relationship.csv")
        log(f"Generating shared device relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]"
            )

        # (src:person)-[:worked_for]->(corp:corporation)<-[:worked_for]-(dst:person)
        _ = "(src:person)-[:worked_for]->(corp:corporation)<-[:worked_for]-(dst:person)"
        log("Generating shared employer relationships in pattern:")
        title("shared employer", (Syntax(_, "cypher", line_numbers=False)))
        # write intermediate data to be column_stacked to the final csv file
        self.csv_writer("data/_shared_employer_relationship.csv",
                        shared_employer_count,
                        self.shared_employer_relationship_generator,
                        index=False)
        shared_employer_rels = pd.read_csv(
            "data/_shared_employer_relationship.csv",
            delimiter=",",
            header=None)
        start_index = shared_phone_num_count + shared_device_count
        end_index = shared_phone_num_count + shared_device_count + shared_employer_count
        concat_shared_employer_rels = pd.concat(
            (edges[start_index:end_index].reset_index(drop=True),
             shared_employer_rels),
            axis=1)
        # header "src_person_id, dst_person_id, corp_id, src_work_for_start_time, dst_work_for_start_time"
        header = [
            "src_id", "dst_id", "corp_id", "src_work_for_start_time",
            "dst_work_for_start_time"
        ]
        _path = "data/shared_employer_relationship.csv"
        concat_shared_employer_rels.to_csv(_path,
                                           sep=",",
                                           index=False,
                                           header=header)
        os.remove("data/_shared_employer_relationship.csv")
        log(f"Generating shared employer relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]"
            )

        # (src:person) -[:worked_for]->(corp:corporation)->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        _ = "(src:person) -[:worked_for]->(corp:corporation)->(pn:phone_number)<-[:with_phone_num]-(dst:person)"
        log("Generating shared phone number and employer relationships in pattern:"
            )
        title("shared phone number and employer",
              Syntax(_, "cypher", line_numbers=False))
        # write intermediate data to be column_stacked to the final csv file
        self.csv_writer("data/_shared_via_employer_phone_num_relationship.csv",
                        shared_via_employer_phone_num_count,
                        self.via_employer_phone_number_relationship_generator,
                        index=False)
        shared_via_employer_phone_num_rels = pd.read_csv(
            "data/_shared_via_employer_phone_num_relationship.csv",
            delimiter=",",
            header=None)
        start_index = shared_phone_num_count + shared_device_count + shared_employer_count
        end_index = shared_phone_num_count + shared_device_count + shared_employer_count + shared_via_employer_phone_num_count
        concat_shared_via_employer_phone_num_rels = pd.concat(
            (edges[start_index:end_index].reset_index(drop=True),
             shared_via_employer_phone_num_rels),
            axis=1)
        # header "src_person_id, dst_person_id, corp_id, phone_num_id, src_work_for_start_time"
        header = [
            "src_id", "dst_id", "corp_id", "pn_id", "src_work_for_start_time"
        ]
        _path = "data/shared_via_employer_phone_num_relationship.csv"
        concat_shared_via_employer_phone_num_rels.to_csv(_path,
                                                         sep=",",
                                                         index=False,
                                                         header=header)
        os.remove("data/_shared_via_employer_phone_num_relationship.csv")
        log(f"Generating shared phone number and employer relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]"
            )

        # (src:person) -[:is_related_to]->(dst:person)
        _ = "(src:person) -[:is_related_to]->(dst:person)"
        log("Generating shared relationship in pattern:")
        title("is related to", Syntax(_, "cypher", line_numbers=False))
        # write intermediate data to be column_stacked to the final csv file
        self.csv_writer("data/_is_relative_relationship.csv",
                        is_relative_count,
                        self.is_related_to_relationship_generator,
                        index=False)
        is_relative_rels = pd.read_csv("data/_is_relative_relationship.csv",
                                       delimiter=",",
                                       header=None)
        start_index = shared_phone_num_count + shared_device_count + shared_employer_count + shared_via_employer_phone_num_count
        end_index = edge_count
        concat_is_relative_rels = pd.concat(
            (edges[start_index:end_index].reset_index(drop=True),
             is_relative_rels),
            axis=1)
        # header "person_id, dst_person_id, degree"
        header = ["person_id", "contact_id", "degree"]
        _path = "data/is_relative_relationship.csv"
        concat_is_relative_rels.to_csv(_path,
                                       sep=",",
                                       index=False,
                                       header=header)
        os.remove("data/_is_relative_relationship.csv")
        log(f"Generating shared relationship ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]"
            )

    def generate_applicants_and_applications_with_is_related_to(self):
        _ = """
        // is related to
        (loan_applicant:appliant) -[:is_related_to]->(contact:person)
        (loan_applicant:appliant) -[:applied_for_loan]->(app:loan_application)
        """
        log("Generating loan application with is_related_to in pattern:")
        console.print(Syntax(_, "cypher", line_numbers=False))

        is_relative_rels = pd.read_csv("data/is_relative_relationship.csv",
                                       delimiter=",")

        # loan_application_count is the row count of is_relative_rels
        loan_application_count = is_relative_rels.shape[0]

        header = [
            "loan_application_id", "address", "degree", "occupation", "salary",
            "is_risky", "risk_profile", "apply_agent_id", "apply_date",
            "application_uuid", "approval_status", "application_type",
            "rejection_reason", "applied_for_loan_start_time"
        ]
        _path = "data/_applicant_application_with_is_related_to.csv"
        self.csv_writer(_path,
                        loan_application_count,
                        self.loan_applicant_and_application_generator,
                        index=True,
                        index_prefix=self.conf["loan_application_id_prefix"] +
                        "r_",
                        header=header)
        applicant_application = pd.read_csv(
            "data/_applicant_application_with_is_related_to.csv",
            delimiter=",")

        # concat applicant_application and is_relative_rels
        concat_is_relative_rels = pd.concat(
            (applicant_application.reset_index(drop=True), is_relative_rels),
            axis=1)

        person = pd.read_csv("data/person.csv", delimiter=",")
        merge_person_cols = pd.merge(concat_is_relative_rels,
                                     person,
                                     on="person_id")
        # transform src_person_id to src_applicant_id
        merge_person_cols.rename(columns={"person_id": "applicant_id"},
                                 inplace=True)
        if self.person_id_prefix != self.applicant_id_prefix:
            merge_person_cols["applicant_id"] = merge_person_cols[
                "applicant_id"].str.replace(self.person_id_prefix,
                                            self.applicant_id_prefix)

        os.remove(_path)
        _path = "data/applicant_application_with_is_related_to.csv"
        merge_person_cols.to_csv(_path, sep=",", index=False)

        log(f"Generating loan application with is_related_to ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]"
            )

    def generate_applicants_and_applications_two_peers(self, pattern_str,
                                                       pattern_name):
        _ = pattern_str
        log(f"Generating loan application with {pattern_name} in pattern:")
        console.print(Syntax(_, "cypher", line_numbers=False))

        relations = pd.read_csv(f"data/{pattern_name}_relationship.csv",
                                delimiter=",")

        # loan_application_count is the row count of relations
        loan_application_count = relations.shape[0]

        header = [
            "loan_application_id", "address", "degree", "occupation", "salary",
            "is_risky", "risk_profile", "apply_agent_id", "apply_date",
            "application_uuid", "approval_status", "application_type",
            "rejection_reason", "applied_for_loan_start_time"
        ]
        header_0 = [h + "_0" for h in header]
        header_1 = [h + "_1" for h in header]
        _path_0 = f"data/_applicant_application_with_{pattern_name}_0.csv"
        _path_1 = f"data/_applicant_application_with_{pattern_name}_1.csv"
        self.csv_writer(_path_0,
                        loan_application_count,
                        self.loan_applicant_and_application_generator,
                        index=True,
                        index_prefix=self.conf["loan_application_id_prefix"] +
                        pattern_name[7] + "_",
                        header=header_0)
        self.csv_writer(_path_1,
                        loan_application_count,
                        self.loan_applicant_and_application_generator,
                        index=True,
                        index_prefix=self.conf["loan_application_id_prefix"] +
                        pattern_name[7] + "_",
                        header=header_1,
                        init_index=loan_application_count + 1)
        applicant_application_0 = pd.read_csv(
            f"data/_applicant_application_with_{pattern_name}_0.csv",
            delimiter=",")
        applicant_application_1 = pd.read_csv(
            f"data/_applicant_application_with_{pattern_name}_1.csv",
            delimiter=",")

        # concat applicant_application_0 and relations
        concat_relations_0 = pd.concat(
            (applicant_application_0.reset_index(drop=True), relations),
            axis=1)
        # concat applicant_application_1 and applicant_application_0
        concat_relations = pd.concat(
            (applicant_application_1.reset_index(drop=True),
             concat_relations_0),
            axis=1)

        person = pd.read_csv("data/person.csv", delimiter=",")
        person.rename(columns={
            "person_id": "src_id",
            "name": "name_0",
            "gender": "gender_0",
            "birthday": "birthday_0"
        },
                      inplace=True)
        merge_person_cols_0 = pd.merge(concat_relations, person, on="src_id")
        # transform src_person_id to applicant_id_0
        merge_person_cols_0.rename(columns={"src_id": "applicant_id_0"},
                                   inplace=True)

        person.rename(columns={
            "src_id": "dst_id",
            "name_0": "name_1",
            "gender_0": "gender_1",
            "birthday_0": "birthday_1"
        },
                      inplace=True)
        merge_person_cols_1 = pd.merge(merge_person_cols_0,
                                       person,
                                       on="dst_id")
        # transform src_person_id to applicant_id_0
        merge_person_cols_1.rename(columns={"dst_id": "applicant_id_1"},
                                   inplace=True)
        if self.person_id_prefix != self.applicant_id_prefix:
            merge_person_cols_1["applicant_id_0"] = merge_person_cols_1[
                "applicant_id_0"].str.replace(self.person_id_prefix,
                                              self.applicant_id_prefix)
            merge_person_cols_1["applicant_id_1"] = merge_person_cols_1[
                "applicant_id_1"].str.replace(self.person_id_prefix,
                                              self.applicant_id_prefix)

        os.remove(_path_0)
        os.remove(_path_1)
        _path = f"data/applicant_application_with_{pattern_name}.csv"
        merge_person_cols_1.to_csv(_path, sep=",", index=False)

        log(f"Generating loan application with {pattern_name} ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]"
            )

    def generate_applicants_and_applications_with_shared_device(self):
        _ = """
        (loan_applicant_0:appliant)-[:used_dev]->(:dev)<-[:used_dev]-(loan_applicant_1:appliant)
        (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
        (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
        """
        pattern_name = "shared_device"
        self.generate_applicants_and_applications_two_peers(_, pattern_name)

    def generate_applicants_and_applications_with_shared_phone(self):
        _ = """
        (loan_applicant_0:appliant)-[:with_phone_num]->(:phone_num)<-[:with_phone_num]-(loan_applicant_1:appliant)
        (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
        (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
        """
        pattern_name = "shared_phone_num"
        self.generate_applicants_and_applications_two_peers(_, pattern_name)

    def generate_applicants_and_applications_with_shared_employer(self):
        _ = """
        (loan_applicant_0:appliant)-[:worked_for]->(:corp)<-[:worked_for]-(loan_applicant_1:appliant)
        (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
        (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
        """
        pattern_name = "shared_employer"
        self.generate_applicants_and_applications_two_peers(_, pattern_name)

    def generate_applicants_and_applications_via_employer_phone_num(self):
        _ = """
        (loan_applicant_0:appliant)-[:worked_for]->(:corp)-[:with_phone_num]->(:phone_num)<-[:with_phone_num]-(loan_applicant_1:appliant)
        (loan_applicant_0:appliant)-[:applied_for_loan]->(app_0:loan_application)
        (loan_applicant_1:appliant)-[:applied_for_loan]->(app_1:loan_application)
        """
        pattern_name = "shared_via_employer_phone_num"
        self.generate_applicants_and_applications_two_peers(_, pattern_name)


def gen_person(step):
    gen = FraudDetectionDataGenerator()
    gen.print_conf()
    title(f"[bold blue][ Step {step} ] [/bold blue]",
          "Generate contacts(person) as vertices")
    gen.generate_contacts()


def gen_homogeneous_rel_with_community(step):
    gen = FraudDetectionDataGenerator()
    title(
        f"[bold blue][ Step {step} ] [/bold blue]",
        "Run ABCD sample to generate relationship data with community structure"
    )
    gen.init_julia()
    gen.run_abcd_sample(gen.abcd_data_dir)


def break_homogeneous_rel_into_heterogeneous_graph_with_community(step):
    gen = FraudDetectionDataGenerator()
    title(f"[bold blue][ Step {step} ] [/bold blue]",
          "Distribute relationships to different patterns")
    gen.generate_clusterred_contacts_relations()


def generate_applicants_and_applications_with_is_related_to(step):
    gen = FraudDetectionDataGenerator()
    title(f"[bold blue][ Step {step} ] [/bold blue]")
    gen.generate_applicants_and_applications_with_is_related_to()


def generate_applicants_and_applications_with_shared_device(step):
    gen = FraudDetectionDataGenerator()
    title(f"[bold blue][ Step {step} ] [/bold blue]")
    gen.generate_applicants_and_applications_with_shared_device()


def generate_applicants_and_applications_with_shared_phone(step):
    gen = FraudDetectionDataGenerator()
    title(f"[bold blue][ Step {step} ] [/bold blue]")
    gen.generate_applicants_and_applications_with_shared_phone()


def generate_applicants_and_applications_with_shared_employer(step):
    gen = FraudDetectionDataGenerator()
    title(f"[bold blue][ Step {step} ] [/bold blue]")
    gen.generate_applicants_and_applications_with_shared_employer()


def generate_applicants_and_applications_via_employer_phone_num(step):
    gen = FraudDetectionDataGenerator()
    title(f"[bold blue][ Step {step} ] [/bold blue]")
    gen.generate_applicants_and_applications_via_employer_phone_num()


if __name__ == "__main__":

    with Progress() as progress:
        task = progress.add_task("[cyan]Progress:", total=5)

        gen = FraudDetectionDataGenerator()
        with Pool(processes=gen.process_count) as pool:

            title(
                "[bold blue][ Init ] [/bold blue]",
                f"Will be running with maximum {gen.process_count} processes")

            step_0 = pool.map_async(gen_person, (0, ))
            progress.advance(task)

            # step 1 which calls PyJulia to be run in main process
            gen_homogeneous_rel_with_community(1)
            progress.advance(task)
            step_2 = pool.map_async(
                break_homogeneous_rel_into_heterogeneous_graph_with_community,
                (2, ))
            step_0.wait()
            progress.advance(task)
            step_2.wait()
            progress.advance(task)

            title(f"[bold blue][ Step 3 ] [/bold blue]",
                  "Generate applicant and applications")
            step_3 = []
            step_3.append(
                pool.map_async(
                    generate_applicants_and_applications_with_is_related_to,
                    (3.0, )))

            step_3.append(
                pool.map_async(
                    generate_applicants_and_applications_with_shared_device,
                    (3.1, )))

            step_3.append(
                pool.map_async(
                    generate_applicants_and_applications_with_shared_phone,
                    (3.2, )))

            step_3.append(
                pool.map_async(
                    generate_applicants_and_applications_with_shared_employer,
                    (3.3, )))

            step_3.append(
                pool.map_async(
                    generate_applicants_and_applications_via_employer_phone_num,
                    (3.4, )))

            for step in step_3:
                step.wait()

        tree = Syntax(OUTPUT_EXPLANATION,
                      "bash",
                      theme="monokai",
                      line_numbers=False)
        title("[bold blue][ Generated Files ] [/bold blue]", tree)
        progress.advance(task)
