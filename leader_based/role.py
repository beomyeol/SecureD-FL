from __future__ import absolute_import, division, print_function

import torch
import io

from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


rho = .01
nIter = 20
tolerance = 0.01

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
        self.totIter = 0

    def begin(self, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        msg_bytes = buffer.getvalue()
        _LOGGER.debug('[leader (%d)] broadcast model parameters. size=%d',
                      self.rank, len(msg_bytes))
        self.network_mgr.broadcast(msg_bytes)

    def end(self, model):
        global rho
        global nIter
        global tolerance
        rhol = rho
        #_LOGGER.info('[leader] before avg model param=%s', str(list(model.named_parameters())))
        
        lambda_dict = {}
        for name, param in model.named_parameters():
            lambda_dict[name] = torch.rand(param.shape)

        x = {}
        z = {}
        z_prv={}
        for name, param in model.named_parameters():
            z[name] = param
            #z[name] = torch.zeros(param.shape)
          
        for i in range(nIter):
            
            for name, param in model.named_parameters():
                x[name] = 1/(2+rhol)*(2*param - lambda_dict[name] + 2*rhol*z[name])    
            for name, param in model.named_parameters():
                z[name] = x[name] + 1.0/rhol*lambda_dict[name]         

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
            torch.save(z, buffer)
            msg_bytes = buffer.getvalue()
            _LOGGER.debug('[leader (%d)] broadcast model parameters. size=%d',
                      self.rank, len(msg_bytes))
            self.network_mgr.broadcast(msg_bytes) 
        
            for name, param in model.named_parameters():
                lambda_dict[name] = lambda_dict[name] + rhol * (x[name] - z[name])
            if i%2==0:
                rhol = rhol/2
                
            dis = 0.0  
            if i>0:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        dis = dis + torch.norm(z[name]-z_prv[name]) 
                    _LOGGER.info('Distance: %s', str(dis))
                     
            if i>0 and dis <= tolerance:
                print(i)
                break

            for name, param in model.named_parameters():
                z_prv[name] = z[name]
                
        model.load_state_dict(z)
        self.totIter += i
        
        #_LOGGER.info('[leader] after avg model param=%s', str(list(model.named_parameters())))

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
        global rho
        global nIter
        global tolerance
        rhol = rho
        lambda_dict = {}
        for name, param in model.named_parameters():
            lambda_dict[name] = torch.rand(param.shape)
            
        z = {}
        z_prv = {}
        for name, param in model.named_parameters():
            #z[name] = torch.zeros(param.shape)
            z[name] = param
            
        x = {}
        dis = 0.0 
        for i in range(nIter):
            for name, param in model.named_parameters():
                x[name] = 1/(2+rhol)*(2*param - lambda_dict[name] + 2*rhol*z[name])

            #send x+1/rho*lambda to the leader
            #prepare what to send
            x_send ={}

            for name, param in model.named_parameters():
                x_send[name] = x[name]+1/rhol*lambda_dict[name]

            buffer = io.BytesIO()
            torch.save(x_send, buffer)
            self.network_mgr.send(self.leader_rank, buffer.getvalue())
 
    
            #TODO
            #recv z from the leader
            msg_bytes = self.network_mgr.recv()
            _LOGGER.debug('[follower (%d)] received model msg. len=%d',
                      self.rank, len(msg_bytes))
            z = torch.load(io.BytesIO(msg_bytes))
            
            for name, param in model.named_parameters():
                lambda_dict[name] = lambda_dict[name] + rhol*(x[name]-z[name])
            
            if i%2==0:
                rhol = rhol/2
                
            dis = 0.0 
            if i>0:
                with torch.no_grad():                    
                    for name, param in model.named_parameters():
                        dis = dis + torch.norm(z[name]-z_prv[name])   
            #print(dis)        
            if i>0 and dis <= tolerance:
                break
                
            for name, param in model.named_parameters():
                z_prv[name] = z[name]
        #update the model?
        model.load_state_dict(z)


    def terminate(self):
        self.network_mgr.terminate()
