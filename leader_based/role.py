from __future__ import absolute_import, division, print_function

import torch
import io

from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class Role(object):

    def begin(self, model):
        pass

    def end(self, model):
        pass

    def terminate(self):
        pass

'''
class Leader(Role):

    def __init__(self, rank, network_mgr):
        self.rank = rank
        self.network_mgr = network_mgr

    def begin(self, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        msg_bytes = buffer.getvalue()
        _LOGGER.debug('[leader (%d)] broadcast model parameters. size=%d',
                      self.rank, len(msg_bytes))
        self.network_mgr.broadcast(msg_bytes)

    def end(self, model):
        accumulated_state_dicts = model.state_dict()
        num_msgs = 1
        while num_msgs < len(self.network_mgr.cluster_spec):
            buffer = io.BytesIO(self.network_mgr.recv())
            for name, tensor in torch.load(buffer).items():
                accumulated_state_dicts[name] += tensor
            num_msgs += 1
            _LOGGER.debug('[leader (%d)] #models=%d', self.rank, num_msgs)

        for tensor in accumulated_state_dicts.values():
            tensor /= num_msgs

        model.load_state_dict(accumulated_state_dicts)

    def terminate(self):
        self.network_mgr.terminate()


class Follower(Role):

    def __init__(self, rank, leader_rank, network_mgr):
        self.rank = rank
        self.leader_rank = leader_rank
        self.network_mgr = network_mgr

    def begin(self, model):
        msg_bytes = self.network_mgr.recv()
        _LOGGER.debug('[follower (%d)] received model msg. len=%d',
                      self.rank, len(msg_bytes))
        model.load_state_dict(torch.load(io.BytesIO(msg_bytes)))

    def end(self, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        self.network_mgr.send(self.leader_rank, buffer.getvalue())
        _LOGGER.debug('[follower (%d)] sent local model', self.rank)

    def terminate(self):
        self.network_mgr.terminate()
'''

class ADMMLeader(Role):

    def __init__(self, rank, network_mgr):
        self.rank = rank
        self.network_mgr = network_mgr

    def begin(self, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        msg_bytes = buffer.getvalue()
        _LOGGER.debug('[leader (%d)] broadcast model parameters. size=%d',
                      self.rank, len(msg_bytes))
        self.network_mgr.broadcast(msg_bytes)

    def end(self, model):
        

        rho = 0.5
        nIter = 30
        lambda_dict = {}
        for name, param in model.named_parameters():
            lambda_dict[name] = torch.rand(param.shape)

        x = {}
        for i in range(nIter):
            for name, param in model.named_parameters():
                x[name] = 2 * param - lambda_dict[name] + 2 * z[name]

            z = {}
            for name, param in model.named_parameters():
                z[name] = x[name] + 1/rho*lambda_dict[name]         

            num_msgs = 1

    
            
            while num_msgs < len(self.network_mgr.cluster_spec):
                buffer = io.BytesIO(self.network_mgr.recv())
                for name, tensor in torch.load(buffer).items():
                    z[name] += tensor
                num_msgs += 1
                _LOGGER.debug('[leader (%d)] #models=%d', self.rank, num_msgs)

            for tensor in z.values():
                tensor /= num_msgs
            #TODO
            #send out the z to the followers
            buffer = io.BytesIO()
            torch.save(z.state_dict(), buffer)
            msg_bytes = buffer.getvalue()
            _LOGGER.debug('[leader (%d)] broadcast model parameters. size=%d',
                      self.rank, len(msg_bytes))
            self.network_mgr.broadcast(msg_bytes) 
        
            for name, param in model.named_parameters():
                lambda_dict[name] = lambda_dict[name] + rho * (x[name] - z[name])
            if i%4==0:
                rho = rho/2


    def terminate(self):
        self.network_mgr.terminate()

class ADMMFollower(Role):

    def __init__(self, rank, leader_rank, network_mgr):
        self.rank = rank
        self.leader_rank = leader_rank
        self.network_mgr = network_mgr

    def begin(self, model):
        msg_bytes = self.network_mgr.recv()
        _LOGGER.debug('[follower (%d)] received model msg. len=%d',
                      self.rank, len(msg_bytes))
        model.load_state_dict(torch.load(io.BytesIO(msg_bytes)))

    def end(self, model):
        rho = 0.5
        nIter = 30
        lambda_dict = {}
        for name, param in model.named_parameters():
            lambda_dict[name] = torch.rand(param.shape)
        z = {}
        for name, param in model.named_parameters():
            z[name] = torch.zeros(param.shape)
        x = {}

        for i in range(nIter):
            for name, param in model.named_parameters():
                x[name] = 2*param - lambda_dict[name] + 2*z[name]

            #send x+1/rho*lambda to the leader
            #prepare what to send
            x_send ={}

            for name, param in model.named_parameters():
                x_send[name] = x[name]+1/rho*lamda_dict[name]

            buffer = io.BytesIO()
            torch.save(x_send.state_dict(), buffer)
            self.network_mgr.send(self.leader_rank, buffer.getvalue())
 
    
            #TODO
            #recv z from the leader
            msg_bytes = self.network_mgr.recv()
            _LOGGER.debug('[follower (%d)] received model msg. len=%d',
                      self.rank, len(msg_bytes))
            z.load_state_dict(torch.load(io.BytesIO(msg_bytes)))
            
            for name, param in model.named_parameters():
                lambda_dict[name] = lambda_dict[name] + rho*(x[name]-z[name])
            
            if i%4==0:
                rho = rho/2


        #update the model?
        model.load_state_dict(z)

    def terminate(self):
        self.network_mgr.terminate()
