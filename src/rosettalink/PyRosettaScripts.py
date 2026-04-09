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

    def init(self, options):
        import pyrosetta
        pyrosetta.init(options)
        self._initialized = True
        self._register_all_components()

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
