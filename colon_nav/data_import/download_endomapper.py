
""""
Step 1: Create user account in Synapse (https://www.synapse.org/#!RegisterAccount:0)
Step 2: Get access to the Endomapper dataset (https://www.synapse.org/#!Synapse:syn26707219)
Step 3: Install required packages: pip install  pysftp pandas synapseclient
Step 4: Run the following code with your Synapse username and password and the path to save the dataset.
""""
import synapseclient
import synapseutils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--synapse_username",
    type=str,
    help="Synapse username",
    default="",
)
parser.add_argument(
    "--synapse_password",
    type=str,
    help="Synapse password",
    default="",
)
parser.add_argument(
    "--save_path",
    type=str,
    help="The path to save the prepared dataset",
    default="",
)
args = parser.parse_args()
save_path = Path(args.save_path)
assert save_path.parent.exists(), "The parent folder of the save path does not exist"
synapse_username = args.synapse_username
synapse_password = args.synapse_password

syn = synapseclient.Synapse()
syn.login(synapse_username, synapse_password)
# download files
entity = syn.get('syn26707219', downloadLocation=save_path)
