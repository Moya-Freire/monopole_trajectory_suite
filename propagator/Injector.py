import concurrent.futures
from InjectorFunctions import *
import h5py
import warnings

################################################################################
#                                                                              #
#                               Injector class                                 #
#                                                                              #
#        Propagates monopoles in bulk with an object-oriented interface        #
#         and saves the results to file, allowing for reproducibility.         #
#                                                                              #
################################################################################


class Injector:
    """
    Object-oriented interface for propagating monopoles in bulk through 
    a given simulation medium. Allows for saving results to file for reproducibility.
    """

    def __init__(self,SimSpace,field,masses,charges,betas,
                 monopoles=None,positions=None,directions=None):
        
        # The injector is expensive to run, very. For good measure, we check if all 
        # information has been given in an usable format before we attempt to do anything
        # with it.

        self.simvolume=SimSpace
        self.field=field
        self.datasets=[]

        # The injector can take the monopoles variable as either an array of monopoles 
        # or an integer representing the number of monopoles to be randomly generated 
        # for the simulation volume in question.

        if isinstance(monopoles, list):
            self.monopoles=monopoles

        elif isinstance(monopoles,int):
            self.monopoles=generate_monopoles(SimSpace,monopoles,betas,masses,
                                              charges,positions,directions)
        elif monopoles is None:
            self.monopoles=[]

        else:
            raise ValueError("Please provide either the number of monopoles " \
            "to be generated or a list with the pre-defined monopoles.")

        # Metadata is tracked for the eventual process of saving the propagation data
        # to file, allowing to track relevant information at run level more easily.

        self.metadata={"Generation datetime":metadata_datetime(),
                        "Run size":len(self.monopoles),
                     }

        self.history_saved=False


    def addMonopoles(self,monopoles,masses,charges,betas,positions=None,directions=None):

        # This method makes it possible for the user to create an injector without any 
        # monopoles and posteriorly add them, or add more monopoles to the injector after
        # propagation (not that it is recommended)

        if isinstance(monopoles,list) or isinstance(monopoles,np.ndarray):
            self.monopoles.extend(monopoles)
        elif isinstance(monopoles,int):
            self.monopoles.extend(generate_monopoles(self.simvolume,monopoles,betas,
                                                     masses,charges,positions,directions))
        else:
            raise ValueError("Please provide either the number of monopoles to be created "
            "or an array of monopoles to be added to the injector")

        # When changes are made to the injector we save the datetime as a snapshot of what
        # stage of the study we were at.

        self.metadata["Run size"]=len(self.monopoles)
        self.metadata["Generation datetime"]=metadata_datetime()


    def propSequential(self,chunk=0.1,condition=True,save_history=True):

        # Multiprocessing might not always be the best choice for every user because 
        # the process of creating multiple processes makes copies of the monopole 
        # objects in the memory that may get heavy and adds an overhead. For small samples
        # and executions in weaker computers, doing the propagation sequentially is possible

        # Guard to track whether the history is being saved for an eventual saving to file.

        if save_history:
            self.history_saved=True

        for id in range(len(self.monopoles)):

            monopole,dataset=propagate_one(self.monopoles[id],id,self.field,
                          self.simvolume,chunk,save_history,condition)
            self.monopoles[id]=monopole
            self.datasets.append(dataset)

        # After the monopole array is updated, we update the metadata datetime so the
        # user can track the chages done to the Injector object

        self.metadata["Generation datetime"]=metadata_datetime()

    def propParallel(self,chunk=0.1,workers=None,condition=True,save_history=True):

        # On windows, doing multiprocessing is a bit finnicky. As a result, when 
        # calling this method for parallel propagation, it is necessary to use the 
        # "if __name__ == '__main__':" guard.

        if save_history:
            self.history_saved=True

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            
            futures = [executor.submit(
                propagate_one, 
                monopole ,
                id,
                self.field,
                self.simvolume,
                chunk,
                save_history,condition) 
                for id,monopole in enumerate(self.monopoles)
            ]

            for future in concurrent.futures.as_completed(futures):
                monopole,dataset=future.result()
                self.monopoles[dataset["ID"]]=monopole
                self.datasets.append(dataset)

        self.metadata["Generation datetime"]=metadata_datetime()

        # We can't quite know in what order the multiprocessing will give us the monopoles 
        # back, as the execution time will be different based on direction, amount of 
        # material traversed, etc. For this reason we reorder our datasets after execution
        # is done to make sure everything is saved to file in order.

        self.datasets.sort(key=lambda d: d["ID"])

    # Saves the data regarding the propagated monopoles to an HDF5 file with compression 
    # to make it more manageable, with the choice to save or not the properties of the 
    # monopole at every propagation step. 

    # IMPORTANT : For a couple hundred thousand monopoles, saving
    # the history will make the process of opening the file a bit unstable. 
    # It is not advised

    def saveState(self,filename="propagated_monopoles.h5",save_history=True):

        if save_history and not self.history_saved:

            # If the user does not save the history during propagation but calls 
            # the save state method with the default signature, it won't fail,
            # but will give them a warning. 

            save_history=False
            warnings.warn("Cannot save history if it wasn't tracked"
                                "during propagation! Saving without history.",RuntimeWarning)

        # While this is not the most elegant way to write a file I/O, I hate making
        # this sort of stuff and value readabilty over anything else when it comes to 
        # it. This might be changed in the future in order to make it less awful.

        with h5py.File(filename,"w") as f:

            for key, value in self.metadata.items():

                f.attrs[key] = value

            for dataset in self.datasets:

                # Each monopole is saved as a group, allowing us to have subcategories
                # within the HDF5 file for them.

                monopole=f.create_group(f"monopole_{dataset["ID"]:06d}")

                # Initial and final information, as well as higher level information
                # such as the numerical ID used for tracking the monopoles and 
                # whether it remained contained within the simulation volume are 
                # all saved as attributes of the group. 

                monopole.attrs["ID"] = dataset["ID"]
                monopole.attrs["Mass"] = dataset["Mass"]
                monopole.attrs["Charge"] = dataset["Charge"]
                monopole.attrs["Contained"] = bool(dataset["Contained"])

                monopole.attrs["Initial position"] = np.asarray(dataset ["Initial position"],
                                                                    dtype = float)
                monopole.attrs["Final position"] = np.asarray (dataset ["Final position"],
                                                                dtype = float)

                monopole.attrs["Initial direction"] = np.asarray(dataset ["Initial direction"], 
                                                                    dtype = float)
                monopole.attrs["Final direction"] = np.asarray (dataset ["Final direction"],
                                                                dtype = float)

                monopole.attrs["Initial velocity"] = dataset["Initial velocity"]
                monopole.attrs["Final velocity"] = dataset["Final velocity"]

                monopole.attrs["Magnetic work"] = dataset["Magnetic work"]
                monopole.attrs["Stopping power work"] = dataset["Stopping power work"]

                # On the other hand, it might be valuable to keep track of how the different
                # characteristics of the monopole change for each propagation step, which 
                # creates arrays of variable size containing these values. These are saved as
                # datasets instead.

                if save_history:

                    monopole.create_dataset("Position history",
                                            data = dataset["Position history"],
                                            compression = "gzip",
                                            compression_opts=4)

                    monopole.create_dataset("Direction history",
                                            data = dataset["Direction history"],
                                            compression = "gzip",
                                            compression_opts=4)

                    monopole.create_dataset("Energy history",
                                            data = dataset["Energy history"],
                                            compression = "gzip",
                                            compression_opts=4)

                    monopole.create_dataset("Velocity history",
                                            data = dataset["Velocity history"],
                                            compression = "gzip",
                                            compression_opts=4)

                    monopole.create_dataset("Magnetic work history",
                                            data = dataset["Magnetic work history"],
                                            compression = "gzip",
                                            compression_opts=4)

                    monopole.create_dataset("Stopping power work history",
                                            data = dataset["Stopping power work history"],
                                            compression = "gzip",
                                            compression_opts=4)