import csv
import os
import re
import julia
import toml

from collections import OrderedDict
from faker import Faker
from julia import Main
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

def title(title, description):
    table = Table(show_header=True)
    table.add_column(title, style="dim", width=96)
    table.add_row(description)
    console.print("\n", table)

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
        self.corporation_id_prefix = self.conf["corporation_id_prefix"]
        Path("data").mkdir(parents=True, exist_ok=True)

    def get_conf(self):
        with open(self.conf_path) as conf_file:
            conf = conf_file.read()
            toml_highlight = Syntax(conf, "toml", theme="monokai", line_numbers=False)
            title("Getting [bold magenta]config.toml[/bold magenta] parsed", toml_highlight)
            return toml.loads(conf)

    def init_julia(self):
        """
        Initialize Julia environment
        """
        julia.install()
        return julia.Julia()

    def run_abcd_sample(self, data_path):
        """
        Run ABCD sample.

        data_path: path to data
        """
        # Create data directory if not exists
        Path(data_path).mkdir(parents=True, exist_ok=True)
        # Run ABCD sample to generate Relationship data with community structure
        log(f"Calling [bold magenta]{ABCD_SAMPLER_PATH}[/bold magenta] to generate community structured data...")
        Main.include(ABCD_SAMPLER_PATH)
        log(f"Calling [bold magenta]{ABCD_SAMPLER_PATH}[/bold magenta]...[green]✓[/green]. Data generated at [bold magenta]{self.abcd_data_dir}[/bold magenta]")

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
                re.escape(self.faker.address().replace("\n", " ").replace(",", " ")),
                is_risky, risk_profile)

    def loan_applicant_generator(self):
        """
        properties: (address, degree, occupation, salary, is_risky, risk_profile)
        """
        is_risky = self.faker.boolean(chance_of_getting_true=float(
            self.conf["chance_of_applicant_risky_percentage"]))
        risk_profile = "NA" if not is_risky else self.faker.sentence()
        return (re.escape(self.faker.address().replace("\n", " ").replace(",", " ")),
                self.faker.random_element(
                    elements=["Bachelor", "Master", "PhD"]), re.escape(self.faker.job().replace(",", " ")),
                self.faker.random_number(digits=6), is_risky, risk_profile)

    def loan_application_generator(self):
        """
        properties: (apply_agent_id, apply_date, application_id, approval_status, application_type, rejection_reason)
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
        (src:loan_applicant:person) -[:applied_for_loan]->(app:loan_application)
        """
        appliant_id = f"{self.person_id_prefix}{self.faker.random_number() % self.person_count + 1}"
        appliant = self.loan_applicant_generator()
        application = self.loan_application_generator()
        start_date = application[1]
        return (appliant_id, ) + appliant + application + (start_date, )

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
        log(f"Generating contacts...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]")

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
        log(f"Generating phones numbers...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]")

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
        log(f"Generating devices...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]")

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
        log(f"Generating corporations...[green]✓[/green]. Data generated at: [bold green]{path}[/bold green]")

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
        title(
            "shared a phone number",
            Syntax(_, "cypher", line_numbers=False))
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
        log(f"Generating shared phone number relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]")

        # (src:person)-[:used_device]->(d:device)<-[:used_device]-(dst:person)
        _ = "(src:person)-[:used_device]->(d:device)<-[:used_device]-(dst:person)"
        log("Generating shared device relationships in pattern:")
        title(
            "shared a device",
            Syntax(_, "cypher", line_numbers=False))
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
        log(f"Generating shared device relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]")

        # (src:person)-[:worked_for]->(corp:corporation)<-[:worked_for]-(dst:person)
        _ = "(src:person)-[:worked_for]->(corp:corporation)<-[:worked_for]-(dst:person)"
        log("Generating shared employer relationships in pattern:")
        title(
            "shared employer",
            (Syntax(_, "cypher", line_numbers=False)))
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
        concat_shared_employer_rels.to_csv(
            _path,
            sep=",",
            index=False,
            header=header)
        os.remove("data/_shared_employer_relationship.csv")
        log(f"Generating shared employer relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]")

        # (src:person) -[:worked_for]->(corp:corporation)->(pn:phone_number)<-[:with_phone_num]-(dst:person)
        _ = "(src:person) -[:worked_for]->(corp:corporation)->(pn:phone_number)<-[:with_phone_num]-(dst:person)"
        log("Generating shared phone number and employer relationships in pattern:")
        title(
            "shared phone number and employer",
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
        concat_shared_via_employer_phone_num_rels.to_csv(
            _path,
            sep=",",
            index=False,
            header=header)
        os.remove("data/_shared_via_employer_phone_num_relationship.csv")
        log(f"Generating shared phone number and employer relationships ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]")

        # (src:person) -[:is_related_to]->(dst:person)
        _ = "(src:person) -[:is_related_to]->(dst:person)"
        log("Generating shared relationship in pattern:")
        title(
            "is related to",
            Syntax(_, "cypher", line_numbers=False))
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
        # header "src_person_id, dst_person_id, degree"
        header = ["src_id", "dst_id", "degree"]
        _path = "data/is_relative_relationship.csv"
        concat_is_relative_rels.to_csv(_path,
                                       sep=",",
                                       index=False,
                                       header=header)
        os.remove("data/_is_relative_relationship.csv")
        log(f"Generating shared relationship ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]")

    def generate_applicants_and_applications(self):
        # (src:loan_applicant:person) -[:applied_for_loan]->(app:loan_application)
        _ = "(src:loan_applicant:person) -[:applied_for_loan]->(app:loan_application)"
        log("Generating loan application relationship in pattern:")
        console.print(Syntax(_, "cypher", line_numbers=False))
        header = [
            "loan_application_id", "person_id", "address", "degree",
            "occupation", "salary", "is_risky", "risk_profile",
            "apply_agent_id", "apply_date", "application_id",
            "approval_status", "application_type", "rejection_reason",
            "applied_for_loan_start_time"
        ]
        _path = "data/_applicant_application_relationship.csv"
        self.csv_writer(_path,
                        self.loan_application_count,
                        self.loan_applicant_and_application_generator,
                        index=True,
                        index_prefix=self.conf["loan_application_id_prefix"],
                        header=header)
        applicant_application = pd.read_csv("data/_applicant_application_relationship.csv",
                                       delimiter=",")
        person = pd.read_csv("data/person.csv", delimiter=",")
        final_result = pd.merge(applicant_application, person, on="person_id")
        final_result.to_csv("data/applicant_application_relationship.csv",
                            sep=",",
                            index=False)
        os.remove("data/_applicant_application_relationship.csv")

        log(f"Generating loan application relationship ...[green]✓[/green]. Data generated at: [bold green]{_path}[/bold green]")

OUTPUT_EXPLANATION = """
$ tree
data
├── abcd             # raw data with ABCD Sampler, reference only
│   ├── com.dat      # vertex -> community
│   ├── cs.dat       # community size
│   ├── deg.dat      # vertex degree
│   └── edge.dat     # edges(which construct the community)
├── applicant_application_relationship.csv
│                    # app vertex and person-applied-> app edge
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
                     # (:p)-[:worked_for]->(:corp)->(:phone_num)<-[:with_phone_num]-(:p)
"""

if __name__ == "__main__":

    with Progress() as progress:

        task = progress.add_task("[cyan]Progress:", total=5)

        gen = FraudDetectionDataGenerator()
        title("[bold blue][ Step 0 ] [/bold blue]", "Generate contacts(person) as vertices")
        progress.advance(task)
        gen.generate_contacts()
        title("[bold blue][ Step 1 ] [/bold blue]", "Run ABCD sample to generate relationship data with community structure")

        progress.advance(task)
        gen.init_julia()
        gen.run_abcd_sample(gen.abcd_data_dir)
        progress.advance(task)

        title("[bold blue][ Step 2 ] [/bold blue]", "Distribute relationships to different patterns")
        gen.generate_clusterred_contacts_relations()
        progress.advance(task)

        title("[bold blue][ Step 3 ] [/bold blue]", "Generate applicant and applications")
        gen.generate_applicants_and_applications()

        tree = Syntax(OUTPUT_EXPLANATION, "bash", theme="monokai", line_numbers=False)
        title("[bold blue][ Generated Files ] [/bold blue]", tree)
        progress.advance(task)
