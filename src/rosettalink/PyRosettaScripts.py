# @file PyRosettaScripts.py
# @brief Setup to integrate external components into RosettaScripts.
# @author Moritz Ertelt, adapted from code written by Samuel Schmitz

class PyRosettaScripts:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PyRosettaScripts, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.mover_creators_ = list()
        self.metric_creators_ = list()
        self.taskOP_creators_ = list()
        self.residue_selector_creators_ = list()
        self._initialized = True

    def init(self, options='-ex1 -ex2aro', *, extra_options='', set_logging_handler=None, notebook=None, silent=False, config=None):
        import pyrosetta
        pyrosetta.init(
            options,
            extra_options=extra_options,
            set_logging_handler=set_logging_handler,
            notebook=notebook,
            silent=silent,
        )
        self._initialized = True
        self._load_configuration(config)
        self._register_all_components()

    @staticmethod
    def _load_configuration(config):
        from .utils import CONFIG_FILENAME, config_search_paths, load_configuration, pose_buffer

        # Poses left over from an earlier protocol in the same process are not
        # inputs to the next one.
        pose_buffer.clear()

        loaded_from = load_configuration(config)
        if loaded_from:
            print(f"Loaded RosettaLink configuration from {loaded_from}")
        else:
            searched = ", ".join(str(path) for path in config_search_paths())
            print(
                f"No {CONFIG_FILENAME} found ({searched}); "
                f"every mover needs run_command set on its tag"
            )

    @staticmethod
    def _register_all_components():
        from .centralized_registration import register_all
        register_all()

    def register_movers(self, movers):

        if not movers:
            return

        from pyrosetta.rosetta.protocols.moves import MoverFactory

        factory = MoverFactory.get_instance()

        for MoverCreator in movers:
            creator = MoverCreator()
            factory.factory_register(creator)
            self.mover_creators_.append(creator)

    def register_metrics(self, metrics):

        if not metrics:
            return

        from pyrosetta.rosetta.core.simple_metrics import SimpleMetricFactory
        factory = SimpleMetricFactory.get_instance()

        for MetricCreator in metrics:
            creator = MetricCreator()
            factory.factory_register(creator)
            self.metric_creators_.append(creator)

    def register_taskops(self, taskops):

        if not taskops:
            return

        from pyrosetta.rosetta.core.pack.task.operation import TaskOperationFactory
        factory = TaskOperationFactory.get_instance()

        for TaskCreator in taskops:
            creator = TaskCreator()
            factory.factory_register(creator)
            self.taskOP_creators_.append(creator)

    def register_residue_selectors(self, residue_selectors):

        if not residue_selectors:
            return

        from pyrosetta.rosetta.core.select.residue_selector import ResidueSelectorFactory
        factory = ResidueSelectorFactory.get_instance()

        for ResidueSelectorCreator in residue_selectors:
            creator = ResidueSelectorCreator()
            factory.factory_register(creator)
            self.residue_selector_creators_.append(creator)

    @staticmethod
    def description():
        return "Register external PyRosettaScripts components with PyRosetta"


pyrosetta_scripts = PyRosettaScripts()
