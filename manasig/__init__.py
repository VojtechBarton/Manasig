
import glob
import os
import random

import h5py
import pickle
import numpy as np
import tempfile
import subprocess

import pandas as pd
from Bio.Seq import Seq
from scipy.io import savemat
from Bio.Blast.Applications import NcbiblastnCommandline, NcbimakeblastdbCommandline
from io import StringIO

#######################
"""Classes to fast5 read format manipulation"""
#######################


class Fast5read:
    """The class that represents one read"""

    def __init__(self, read_id, fast5):
        """
        Construction of the read.

        :param read_id: The name of the read as in f5 file
        :param fast5: The read as h5py object
        """
        self._read_id = read_id
        self._fast5 = fast5

    def get_name(self):
        """
        Get name of the read.

        :return: Name of the read as str
        """
        return self._read_id

    def get_raw(self):
        """
        Get raw read signal as numpy array.

        :return: Raw read signal as np.array
        """
        return np.asarray(self._fast5['Raw/Signal'])

    def get_signal(self):
        """
        Get signal recomputed to pA.

        :return: Read recomputed squiggle as np.array
        """
        channel_range = self._fast5['channel_id'].attrs['range']
        digitisation = self._fast5['channel_id'].attrs['digitisation']
        offset = self._fast5['channel_id'].attrs['offset']

        raw_unit = channel_range / digitisation
        signal = (self.get_raw() + offset) * raw_unit

        return signal

    def get_fastq(self, group='000', outfile=None):
        """
        Get fastq of the read.

        :param group: Read group id
        :param outfile: file to write output to
        :return: Read as fastq string
        """
        path = 'Analyses/Basecall_1D_{group}/BaseCalled_template/Fastq'.format(group=group)
        fastq = self._fast5[path][()].decode('UTF-8')

        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(fastq)

        return fastq

    def get_fasta(self, group='000', outfile=None):
        """
        Get fasta of the read

        :param group: Read group id
        :param outfile: file to write output to
        :return: Read as fasta string
        """
        fasta = '>' + self.get_name() + '\n' + self.get_seq(group)

        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(fasta)

        return fasta

    def get_seq(self, group='000'):
        """
        Get sequence of the basecalled read.

        :param group:  Read group id
        :return: Sequence of the read as str
        """
        return self.get_fastq(group).split('\n')[1].strip()

    def get_revseq(self, group='000'):
        """
        Get sequence of the basecalled read.

        :param group:  Read group id
        :return: Sequence of the read as str
        """
        return self.get_fastq(group).split('\n')[1].strip()[::-1]

    def get_complement(self, group='000'):
        """
        Get complement sequence of the basecalled read.

        :param group: Read group id
        :return: Sequence complement as str
        """
        seq = Seq(self.get_seq(group))
        complement = seq.complement()

        return str(complement)

    def get_rev_complement(self, group='000'):
        """
        Get reverse complement of the sequence.

        :param group: Read group id
        :return: Reverse complement of the sequence as str
        """
        seq = Seq(self.get_seq(group))
        rev_complement = seq.reverse_complement()

        return str(rev_complement)

    def get_moves(self, group='000'):
        """
        Get the move array as produced by guppy.

        :param group:  Read group id
        :return: Guppy moves from basecalling as np.array
        """
        path = 'Analyses/Basecall_1D_{group}/BaseCalled_template/Move'.format(group=group)
        return np.asarray(self._fast5[path])

    def get_start(self, group='000'):
        """
        Get the position of the start of basecalling.
        :param group:  Read group id
        :return: Start position of the basecalling trace as int
        """
        path = 'Analyses/Segmentation_{group}/Summary/segmentation'.format(group=group)
        return int(self._fast5[path].attrs['first_sample_template'])

    def get_step(self, group='000'):
        """
        Get step parameter from basecalling.

        :param group:  Read group id
        :return: Basecalling step parameter as int
        """
        path = 'Analyses/Basecall_1D_{group}/Summary/basecall_1d_template'.format(group=group)
        return int(self._fast5[path].attrs['block_stride'])

    def get_basepositions(self, group='000'):
        """
        Get the start positions of the basecalls in raw data.

        :param group:  Read group id
        :return: Positions of basecalls as np.array
        """
        moves = self.get_moves(group)
        start = self.get_start(group)
        step = self.get_step(group)
        basepositions = []
        for i in range(len(moves)):
            if moves[i]:
                basepositions.append(start + (step * i))

        return basepositions

    def is_subseq_in(self, subseq, revcompl=True, group='000'):
        """
        Get logical value if the substring is in read sequence

        :param subseq: substring to find
        :param revcompl: if look also in revcomplement
        :param group: Read group id
        :return: True if the substring is present in the sequence
        """

        if subseq in self.get_seq(group):
            return True
        if revcompl:
            if subseq in self.get_rev_complement(group):
                return True
        else:
            return False

    def get_subseq_signal(self, subseq, raw=False, group='000'):
        """
        Get the signal of the corresponding substring from the whole read

        :param subseq: substring to find in the sequence
        :param raw: if the output should be a raw signal, if not the recomputed one is returned
        :param group: Read group id
        :return: signal of corresponding substring as numpy array
        """
        indices = self.get_seq(group).find(subseq)
        if indices == -1:
            indices = self.get_rev_complement(group).find(subseq)
            if indices == -1:
                return 0

        if raw:
            signal = self.get_raw()
        else:
            signal = self.get_signal()

        basepositions = self.get_basepositions(group)
        start = basepositions[indices]
        stop = basepositions[indices + len(subseq)] - 1
        subsignal = signal[start:stop]

        return subsignal

    def read_to_file(self, file):
        """
        Write read to fast5 file.
        :param file: file to write the read into
        """
        if isinstance(file, str):
            fd = h5py.File(file, 'a')
            self._fast5.copy(self._fast5, fd)
            fd.close()
        elif isinstance(file, h5py.File):
            self._fast5.copy(self._fast5, file)
        else:
            raise IOError


class Fast5file:
    """The class that represents one fast5 file containing multiple reads"""

    def __init__(self, path, mode='r'):
        """
        Open the file.

        :param path: The path to the file as string
        :param mode: The mode how the file should be opened
        """
        self.root = h5py.File(path, mode=mode)
        self._run_id = self.root[list(self.root)[0]].attrs['run_id']

    def __iter__(self):
        """
        Iterate over the reads_names in file.

        :return: An iterator over the names of the reads
        """
        for read_name in self.root:
            yield read_name

    def write_fast5_file(self, file, update=False):
        """
        Write Fast5file too a file.
        :param file: path to the file to be saved in.
        :param update: if the object root path should be updated to a new file
        """
        fd = h5py.File(file, mode="w")
        self.root.copy(self.root, fd)
        fd.close()

        if update:
            self.root = h5py.File(file, 'r')

    def get_run_id(self):
        """
        Get run_id.

        :return: run_id
        """
        return self._run_id

    def get_reads_ids_list(self):
        """
        Get all reads ids as a list

        :return: list of all reads ids
        """
        return list(self.root)

    def get_reads_count(self):
        return len(self.get_reads_ids_list())

    def get_read(self, read_id):
        """
        Get specific read from multi read fast5.

        :param read_id: str, name of the read
        :return: Fast5read object of specified read
        """
        return Fast5read(read_id, self.root[read_id])

    def get_fastq(self, group='000', outfile=None):
        """
        Get fastq of the reads.

        :param group: Group read
        :param outfile: file to write output to
        :return: Reads fastq as str
        """
        fastq = str()
        for read_name in self.__iter__():
            read = self.get_read(read_name)
            fastq += '\n'
            fastq += read.get_fastq(group)

        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(fastq)

        return fastq

    def get_fasta(self, group='000', outfile=None):
        """
        Get fasta of the reads

        :param group: Group read id
        :param outfile: file to write output to
        :return: Reads fasta as str
        """
        fasta = str()
        for read_name in self.__iter__():
            read = self.get_read(read_name)
            fasta += read.get_fasta(group)
            fasta += '\n'

        if outfile is not None:
            with open(outfile, 'a') as f:
                f.write(fasta)

        return fasta


class Fast5run:
    """The class that represents one nanopore sequencing run, containing multiple fast5 files"""

    def __init__(self, path, ext='.fast5', blastdb=None):
        """
        Open the folder and find all fast5 files.

        :param path: The path to the folder as string, as "path/" (with slash at the end)
        :param ext: The extension of the fast5 files as string
        """
        self._path = path
        self._extension = ext
        self._files = [os.path.basename(x) for x in glob.glob(os.path.join(path, '*{ext}'.format(ext=ext)))]
        self._run_id = self.get_file(self._files[0]).get_run_id()

        if blastdb is None:
            if os.path.exists(os.path.join(path, 'blastdb/')):
                blastdb = os.path.join(path, 'blastdb/db')
        self._blastdb = blastdb
        self.blast_out = None

    def __iter__(self):
        """
        Iterate over the files in the folder.

        :return: An iterator over the file names
        """
        for file in self._files:
            yield file

    def get_blastdb(self):
        """
        Get a path to blast database

        :return: path to blast database
        """
        return self._blastdb

    def get_reads_count(self):
        count = 0
        for file_name in self.__iter__():
            file = self.get_file(file_name)
            count += file.get_reads_count()
        return count

    def get_file(self, file_name):
        """
        Get specific file from the folder.

        :param file_name: The filename as string
        :return: The specified file as a Fast5file object
        """
        path = os.path.join(self._path, file_name)

        return Fast5file(path)

    def get_random_file(self):
        """
        Get random file from the folder.

        :return: Random file as a Fast5file object
        """
        file_name = random.choice(self._files)

        return self.get_file(file_name)

    def get_fastq(self, group='000', outfile=None):
        """
        Get fastq of whole run.

        :param group: Group read id
        :param outfile: file to write output to
        :return: Reads fastq as str
        """
        fastq = str()
        if outfile is not None:
            with open(outfile, 'w') as f:
                for file_name in self.__iter__():
                    file = self.get_file(file_name)
                    f.write(file.get_fastq(group))
                    f.write('\n')

        else:
            for file_name in self.__iter__():
                file = self.get_file(file_name)
                fastq += file.get_fastq(group) + '\n'

            return fastq

    def get_fasta(self, group='000', outfile=None):
        """
        Get fasta of the whole run.

        :param group: Group read id
        :param outfile: file to write output to
        :return: Reads fasta as str
        """
        fasta = str()
        if outfile is not None:
            with open(outfile, 'w') as f:
                for file_name in self.__iter__():
                    file = self.get_file(file_name)
                    f.write(file.get_fasta(group))
                    f.write('\n')

        else:
            for file_name in self.__iter__():
                file = self.get_file(file_name)
                fasta += file.get_fasta(group) + '\n'

            return fasta

    def make_blastdb(self, output=None, input_fa=None, group='000', **kwargs):
        """
        Make a blast database, for searching

        :param output: path to output the database files, name of the database
        :param input_fa: path to fasta file to create db from, by default takes all the reads in the run
        :param group: Group read id
        """
        if output is None:
            output = os.path.join(self._path, 'blastdb/db')
        if input_fa is None:
            tmp_file, tmp_path = tempfile.mkstemp(suffix='.fasta', text=True)
            self.get_fasta(group, tmp_path)
            input_fa = tmp_path

        cline = NcbimakeblastdbCommandline(input_file=input_fa, input_type='fasta', dbtype='nucl', out=output, **kwargs)
        cmdout, _ = cline()
        # blastn_db = "makeblastdb -in {} -dbtype nucl -input_type fasta -out {}".format(input_fa, output)
        # subprocess.run(blastn_db, shell=True)

        self._blastdb = output

        return cmdout

    def search_blast(self, query: str, **kwargs):
        cline = NcbiblastnCommandline(query=query, db=self.get_blastdb(), outfmt=6, **kwargs)
        out, _ = cline()
        blast_table = pd.read_csv(StringIO(out), sep='\t')
        blast_table.columns = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend',
                               'sstart', 'send', 'evalue', 'bitscore']
        self.blast_out = blast_table

        return blast_table

    def get_blast_table(self):
        return self.blast_out

    def get_reads_by_id(self, reads: list, outfile=None):
        """
        Get selected reads by their id.

        :param reads: list of reads ids
        :param outfile: file, where the selected reads should be written to
        :return: Fast5File - with a search results, file_list - all files, where any of the reads was matched
        """
        if outfile is None:
            tmp_file, tmp_path = tempfile.mkstemp(suffix='.fast5')
            fd = h5py.File(tmp_path, 'a')
        else:
            fd = h5py.File(outfile, 'a')

        reads_set = set(reads)
        for file_name in self.__iter__():
            file = self.get_file(file_name)
            file_reads = set(file.get_reads_ids_list())
            for read_id in file_reads.intersection(reads_set):
                read = file.get_read(read_id)
                read.read_to_file(fd)

        f5file = Fast5file(fd.filename)
        fd.close()

        return f5file

    def get_reads_from_blast(self, blast_table=None, query=None, outfile=None):
        if blast_table is None:
            blast_table = self.get_blast_table().copy()
        if query:
            blast_table = blast_table[blast_table['qseqid'] == query]

        return self.get_reads_by_id(list(blast_table['sseqid']), outfile=outfile)


#######################
"""Functions to fast5 pickle"""
#######################


def write_fast5_obj(obj, outfile):
    """
    Write fast5 obj to a file
    :param obj: object to be written
    :param outfile: file to be written to
    """
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f)


def load_fast5_obj(infile):
    """
    Load obj from file
    :param infile: file with the object
    :return: loaded object
    """
    with open(infile, 'r') as f:
        obj = pickle.load(f)
    return obj


def write_file_to_mat(fast5file: Fast5file, outfile):
    """
    write selected features of reads in fast5file to matlab .mat format
    :param fast5file: Fast5file object, which to convert
    :param outfile: where the .mat file should be written to
    """
    read_id_list = list()
    raw_list = list()
    signal_list = list()
    basepositions_list = list()
    sequence_list = list()

    for read_name in fast5file.__iter__():
        read = fast5file.get_read(read_name)
        read_id_list.append(read.get_name())
        raw_list.append(read.get_raw())
        signal_list.append(read.get_signal())
        basepositions_list.append(read.get_basepositions())
        sequence_list.append(read.get_seq())

    dct = {"read_id": read_id_list,
           "raw": raw_list,
           "signal": signal_list,
           "basepositions": basepositions_list,
           "sequence": sequence_list}

    savemat(outfile, dct)


#######################
"""Functions to fast5 search manipulation"""
#######################


def get_reads_by_id(reads: list, run: Fast5run, outfile=None):
    """
    Get selected reads by their id.

    :param reads: list of reads ids
    :param run: Fast5run object, where to search
    :param outfile: file, where the selected reads should be written to
    :return: Fast5File - with a search results, file_list - all files, where any of the reads was matched
    """
    if outfile is None:
        tmp_file, tmp_path = tempfile.mkstemp(suffix='.fast5')
        fd = h5py.File(tmp_path, 'a')
    else:
        fd = h5py.File(outfile, 'a')

    file_list = list()

    reads_set = set(reads)
    for file_name in run.__iter__():
        file = run.get_file(file_name)
        file_reads = set(file.get_reads_ids_list())
        for read_id in file_reads.intersection(reads_set):
            read = file.get_read(read_id)
            read.read_to_file(fd)
            file_list.append(file_name)

    f5file = Fast5file(fd.filename)
    fd.close()

    return f5file, file_list


def get_reads_with_subseq(subseq: str, run: Fast5run, outfile=None, revcompl=False):
    """
    Get all reads containing specific subsequence

    :param subseq: sequence to be looked for
    :param run: Fast5run object, where to search
    :param outfile: file, where the selected reads should be written to
    :param revcompl: if the reverse complement should be searched too
    :return: Fast5File - with a search results, file_list - all files, where any of the reads was matched
    """
    if outfile is None:
        tmp_file, tmp_path = tempfile.mkstemp(suffix='.fast5')
        fd = h5py.File(tmp_path, 'a')
    else:
        fd = h5py.File(outfile, 'a')

    file_list = list()

    for file_name in run.__iter__():
        file = run.get_file(file_name)
        for read_id in file.get_reads_ids_list():
            read = file.get_read(read_id)
            if read.is_subseq_in(subseq, revcompl):
                read.read_to_file(fd)
                file_list.append(file_name)

    f5file = Fast5file(fd.filename)
    fd.close()

    return f5file, file_list


def blast_search(queryfile: str, run: Fast5run, outfile=None):
    """
    Search in a run by blast

    :param queryfile: path to fasta file with a query sequence
    :param run: Fast5run object, where to search
    :param outfile: path to file, where the output should be written
    """
    blastn_cmd = 'blastn -out {} -outfmt 5 -query {} -db {} -strand both -max_target_seqs 10000'.format(outfile, queryfile, run.get_blastdb())
    subprocess.run(blastn_cmd, shell=True)
