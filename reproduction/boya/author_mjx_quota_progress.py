"""Count valid fit labels separately from failures, internal rows and unused slots."""
from author_expansion_progress import Progress

UNSTARTED={'quota_reached_not_started','budget_not_started','controller_error_or_not_executed'}


class QuotaProgress(Progress):
    def __init__(self,directory,protocol):
        self.specs={s['index']:s for s in protocol['rows']}
        self.existing_fit=protocol['existing_fit_records']
        self.existing_internal=protocol['existing_internal_records']
        self.target=protocol['target_fit_records']
        super().__init__(directory,len(protocol['rows']))

    def counts(self,*,audited):
        result={'fit':0,'internal':0}
        for i,row in self.records.items():
            if not row['eligible'] or i not in self.returns or not self.returns[i]['passed']:continue
            if audited and i not in self.verified:continue
            result[self.specs[i]['split']]+=1
        return result

    def runtime_target_ready(self):
        with self.lock:return self.existing_fit+self.counts(audited=False)['fit']>=self.target

    def summary(self):
        with self.lock:
            result=super().summary();runtime=self.counts(audited=False);audited=self.counts(audited=True)
            processed=[r for r in self.records.values() if r['status']not in UNSTARTED]
            result.update(target_fit_records=self.target,existing_fit_records=self.existing_fit,
                existing_internal_records=self.existing_internal,
                new_returned_fit_records=runtime['fit'],new_audited_fit_records=audited['fit'],
                cumulative_returned_fit_records=self.existing_fit+runtime['fit'],
                cumulative_audited_fit_records=self.existing_fit+audited['fit'],
                cumulative_audited_internal_records=self.existing_internal+audited['internal'],
                remaining_audited_fit_records=max(0,self.target-self.existing_fit-audited['fit']),
                processed_terminal_scenes=len(processed),processed_null_scenes=sum(not r['eligible']for r in processed),
                unstarted_slots=len(self.records)-len(processed),
                fit_target_audited=self.existing_fit+audited['fit']>=self.target,video_recording=False)
            return result
