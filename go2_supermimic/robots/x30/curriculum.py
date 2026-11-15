"""Measured-skill curriculum, independent of the simulator and training loop."""
from copy import deepcopy
import math
import torch

LEVELS={'forward':(.5,.8,1.2,1.8,2.4,3.),'backward':(.2,.3,.45,.6),
        'lateral':(.2,.3,.5,.75,1.),'yaw':(.2,.4,.7,1.,1.2)}
MODES=('stop','forward','backward','lateral','rotation','turn','uturn')
MODE_PROBABILITIES=(.10,.30,.10,.20,.10,.15,.05)
MAX_LATERAL_ACCELERATION=1.5


class Curriculum:
    levels=LEVELS
    version=1
    mode_probabilities=MODE_PROBABILITIES
    probe_weight_defaults={key:.2 for key in LEVELS}
    probe_weight_max=.2

    def __init__(self):
        self.state={'version':self.version,'last_evaluation':0,'axes':{
            key:{'level':0,'passes':0,'failures':0,
                 'probe_weight':self.probe_weight_defaults[key]} for key in self.levels}}

    def state_dict(self):return deepcopy(self.state)

    def load_state_dict(self,state):
        try:
            if state['version']!=self.version or type(state['last_evaluation']) is not int or state['last_evaluation']<0:
                raise ValueError('invalid curriculum version/evaluation')
            if set(state['axes'])!=set(self.levels):raise ValueError('invalid curriculum axes')
            for key,levels in self.levels.items():
                row=state['axes'][key]
                if any(type(row[k]) is not int or row[k]<0 for k in ('level','passes','failures')):
                    raise ValueError('invalid curriculum counters')
                if row['level']>=len(levels) or row['passes']>1 or row['failures']>1:
                    raise ValueError('invalid curriculum level/streak')
                p=row['probe_weight']
                if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not .05<=p<=self.probe_weight_max:
                    raise ValueError('invalid probe probability')
        except (KeyError,TypeError) as exc:raise ValueError('Malformed curriculum state') from exc
        self.state=deepcopy(state)

    def caps(self,probe=False):
        return {k:levels[min(self.state['axes'][k]['level']+int(probe),len(levels)-1)] for k,levels in self.levels.items()}

    def observe(self,report):
        try:
            it=report['iteration']
            if type(it) is not int or it<=self.state['last_evaluation']:raise ValueError('stale evaluation')
            if set(report['axes'])!=set(self.levels):raise ValueError('incomplete evaluation')
            flags=[report['retention_passed'],report['world_model_passed']]
            flags += [report['axes'][k][b] for k in self.levels for b in ('mastered','probe')]
            if any(type(v) is not bool for v in flags):raise ValueError('evaluation gates must be booleans')
        except (KeyError,TypeError) as exc:raise ValueError('Malformed evaluation report') from exc
        events=[];self.state['last_evaluation']=it
        for key,levels in self.levels.items():
            row=self.state['axes'][key];score=report['axes'][key]
            good=score['mastered'] and score['probe'] and report['retention_passed'] and report['world_model_passed']
            row['passes']=row['passes']+1 if good else 0
            row['failures']=row['failures']+1 if not score['mastered'] else 0
            if row['passes']>=2:
                old=row['level'];row['level']=min(old+1,len(levels)-1);row['passes']=0
                row['probe_weight']=self.probe_weight_defaults[key]
                if row['level']!=old:events.append({'axis':key,'decision':'promote','cap':levels[row['level']]})
            elif row['failures']>=2:
                row['level']=max(0,row['level']-1);row['failures']=0;row['probe_weight']=max(.05,row['probe_weight']/2)
                events.append({'axis':key,'decision':'demote','cap':levels[row['level']]})
            elif not score['probe']:
                row['probe_weight']=max(.05,row['probe_weight']*.75)
        return events

    def sample(self,count,generator,device):
        def random():return torch.rand(count,device=device,generator=generator)
        modes=torch.multinomial(torch.tensor(self.mode_probabilities,device=device),count,replacement=True,generator=generator)
        caps=self.caps();probes=self.caps(True)
        def speed(key):
            level=self.state['axes'][key]['level'];old=self.levels[key][max(0,level-1)]
            kind=random();u=random();p=self.state['axes'][key]['probe_weight']
            easy=.2+u*(old-.2)
            practice=max(.2,old)+u*(caps[key]-max(.2,old))
            frontier=caps[key]+u*(probes[key]-caps[key])
            return torch.where(kind<p,frontier,torch.where(kind<p+.5,practice,easy))
        forward,backward,lateral,yaw=(speed(k) for k in self.levels)
        sign=torch.where(random()<.5,-1.,1.)
        command=torch.zeros(count,3,device=device)
        command[:,0]=torch.where((modes==1)|(modes==5),forward,torch.where(modes==2,-backward,0.))
        command[:,1]=torch.where(modes==3,lateral*sign,0.)
        command[:,2]=torch.where((modes==4)|(modes==5)|(modes==6),yaw*sign,0.)
        command[:,2]=torch.where(modes==5,command[:,2].clamp(-MAX_LATERAL_ACCELERATION/forward,MAX_LATERAL_ACCELERATION/forward),command[:,2])
        return command,modes
